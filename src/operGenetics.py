import numpy as np
from scipy.stats import qmc
from src import geoBlade as gb
from src import aeroPropeller as ap
import os
import json
import math

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):  
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def initialPopulation(fileGen, N, Omega):
    D = len(Omega)
    Dh = 0
    var = []
    for j in range(D):
        if Omega[j,0]!=Omega[j,1]:
            Dh += 1
            var.append(j)
    sampler = qmc.LatinHypercube(d=Dh)
    sample = sampler.random(n=N)
    LHx = qmc.scale(sample, Omega[var,0], Omega[var,1])
    Pg = np.zeros((N, D))
    idg = []
    for i in range(N):
        idg.append(fileGen+'/blade_'+str(i))
        k = 0
        for j in range(D):
            if (j in var) == True:
                Pg[i,j] = LHx[i,k]
                k += 1
            else:
                Pg[i,j] = Omega[j,0]
    Pg[:,8] = np.round(Pg[:,8])
    return Pg, idg


def objectiveFunction(N, Pg, nameBlade, scaler, mlp, vae, flow, paramBEMT):
    nS = paramBEMT['sect_body'] + paramBEMT['sect_tip']
    nu = flow['nu']
    rho = flow['rho']
    Vaxial = flow['V_axial']
    Vtrans = flow['V_trans']
    # a = flow['a']
    
    Vinf = np.sqrt(Vaxial**2 + Vtrans**2)
    alphaShaft = np.degrees(np.arctan2(Vtrans, Vaxial))
    
    n_min_u = np.array([])
    n_min_l = np.array([])
    d = np.array([])
    nB = np.array([])
    Z = np.array([])
    DS = np.empty([0,21])
    for i in range(N):
        os.makedirs(nameBlade[i], exist_ok=True)
        dS = gb.paramBlade(Pg[i], nameBlade[i], paramBEMT['rR_min'],
                           paramBEMT['rR_max'], paramBEMT['sect_body'], 
                           paramBEMT['sect_tip'])
        DS = np.row_stack((DS, dS))
        d = np.append(d, Pg[i,9])
        nB = np.append(nB, Pg[i,8])
        n_min_u = np.append(n_min_u, Pg[i,14])
        n_min_l = np.append(n_min_l, Pg[i,19])
        Z = np.append(Z, Pg[i,9]*flow['Z_d'])
        print('Propeller_'+str(i)+' drawn')

    W_g = np.array([])
    T_g = np.array([])
    T_u_g = np.array([])
    T_l_g = np.array([])
    Q_u_g = np.array([])
    Q_l_g = np.array([])
    FM_u_g = np.array([])
    FM_l_g = np.array([])
    phi_u_g = np.array([])
    phi_l_g = np.array([])
    
    for i in range(N):
        ds = DS[nS*i:nS*(i+1)]
        imgPred_i = ap.precomputeAirfoilImages(scaler, mlp, vae, ds[:,:12])
        aeroResults = ap.TWQCoaxial_Oblique_Vectorized(n_min_u[i], n_min_l[i], nB[i], d[i], 
                                                       ds[:,16], ds[:,13], ds[:,14], ds[:,15], 
                                                       imgPred_i, Z[i], rho, nu, 
                                                       V_inf=Vinf, alpha_shaft_deg=alphaShaft)
        
        T, W, _, _, _, Cw, f_wake, T_u, T_l, _, _, Q_u, Q_l, FM_u, FM_l, _, phi_u, phi_l = aeroResults
        print('Propeller_'+str(i)+' calculated')
        if math.isnan(T):
            W_g = np.append(W_g, 100000)
            T_g = np.append(T_g, 100000)
            Q_u_g = np.append(Q_u_g, 100000)
            Q_l_g = np.append(Q_l_g, 100000)
        else:
            W_g = np.append(W_g, W)
            T_g = np.append(T_g, T)
            Q_u_g = np.append(Q_u_g, Q_u)
            Q_l_g = np.append(Q_l_g, Q_l)
        T_u_g = np.append(T_u_g, T_u)
        T_l_g = np.append(T_l_g, T_l)
        FM_u_g = np.append(FM_u_g, FM_u)
        FM_l_g = np.append(FM_l_g, FM_l)
        phi_u_g = np.append(phi_u_g, phi_u)
        phi_l_g = np.append(phi_l_g, phi_u)
    
        info_prop = {}
        info_prop['nameCase'] = nameBlade[i]
        info_prop['Total Thrust'] = T
        info_prop['Thrust (upper)'] = T_u
        info_prop['Thrust (lower)'] = T_l
        info_prop['Total Power'] = W
        info_prop['Torque (upper)'] = Q_u
        info_prop['Torque (lower)'] = Q_l
        info_prop['FM (upper)'] = FM_u
        info_prop['FM (lower)'] = FM_l
        info_prop['Cw'] = Cw
        info_prop['f_wake'] = f_wake
        info_prop['V_axial'] = flow['V_axial']
        info_prop['V_trans'] = flow['V_trans']
        design_param = {}
        design_param['c_d_r'] = Pg[i,0]
        design_param['c_d_m'] = Pg[i,1]
        design_param['c_d_t'] = Pg[i,2]
        design_param['r_cdm'] = Pg[i,3]
        design_param['yt_r'] = Pg[i,4]
        design_param['yt_m'] = Pg[i,5]
        design_param['yt_t'] = Pg[i,6]
        design_param['r_yt'] = Pg[i,7]
        design_param['B'] = Pg[i,8]
        design_param['d'] = Pg[i,9]
        design_param['alpha_r_up'] = Pg[i,10]
        design_param['alpha_m_up'] = Pg[i,11]
        design_param['alpha_t_up'] = Pg[i,12]
        design_param['r_alpham_up'] = Pg[i,13]
        design_param['n_min_up'] = Pg[i,14]
        design_param['alpha_r_low'] = Pg[i,15]
        design_param['alpha_m_low'] = Pg[i,16]
        design_param['alpha_t_low'] = Pg[i,17]
        design_param['r_alpham_low'] = Pg[i,18]
        design_param['n_min_low'] = Pg[i,19]
        
        info_prop['design_parameters'] = design_param
        sections_data = {}
        sections_data['r_R'] = ds[:,12].tolist()
        sections_data['r'] = ds[:,16].tolist()
        sections_data['c'] = ds[:,13].tolist()
        sections_data['alpha_up'] = ds[:,14].tolist()
        sections_data['phi_up'] = phi_u.tolist()
        sections_data['alpha_low'] = ds[:,15].tolist()
        sections_data['phi_low'] = phi_l.tolist()
        sections_data['xt'] = ds[:,17].tolist()
        sections_data['yt'] = ds[:,18].tolist()
        sections_data['xc'] = ds[:,19].tolist()
        sections_data['yc'] = ds[:,20].tolist()
        info_prop['info_sections'] = sections_data
        
        json_object = json.dumps(info_prop, indent=11, cls=NumpyEncoder)      
        with open(nameBlade[i] + '/info_prop.json', 'w') as outfile:
            outfile.write(json_object)
            
    TWQg = np.column_stack((T_g, W_g, Q_u_g, Q_l_g, T_u_g, T_l_g, FM_u_g, FM_l_g))
    
    return TWQg

def distanceX(Wg, Tg, Tmin, Qg_u, Qg_l, Qtol):
    # Normalize functions
    fn = np.array([])
    for i in range(len(Wg)):
        fn = np.append(fn, (Wg[i]-min(Wg))/(max(Wg)-min(Wg)))
    # Constrain functions
    g1 = (Tmin - Tg)/Tmin
    g2 = np.abs(Qg_u - Qg_l) - Qtol
    # Constrain violations
    c1 = np.maximum(0, g1)
    c2 = np.maximum(0, g2)
    c_total = c1 + c2
    
    max_c = max(c_total)
    if max_c > 0:
        v = c_total / max_c
    else:
        v = np.zeros_like(c_total)
    
    rf = len(np.where(v==0)[0])/len(fn)
    
    d = np.zeros_like(fn)
    for i in range(len(fn)):
        if rf == 0:
            d[i] = v[i]
        else:
            d[i] = np.sqrt(fn[i]**2+v[i]**2)
    return fn, v, d

def twoPenalties(fn, v):
    rf = len(np.where(v==0)[0])/len(fn)
    X = np.array([])
    for i in range(len(fn)):
        if rf==0:
            X = np.append(X, 0)
        else:
            X = np.append(X, v[i])
    Y = np.zeros_like(fn)
    for i in range(len(fn)):
        if v[i] == 0:
            Y[i] = 0
        else:
            Y[i] = fn[i]
    p = np.zeros_like(fn)
    for i in range(len(fn)):
        p[i] = (1-rf)*X[i]+rf*Y[i]
    return p

def newSizePop(N0, G, g, alpha=5, Nend=20):
    c = -np.log((Nend-alpha)/(N0+alpha))/G
    Ng = round((N0+alpha)*np.exp(-c*g)-alpha)
    if Ng < Nend:
        Ng = Nend
    if Ng%2 == 0:
        NE = 2
    else:
        NE = 3
    return Ng, NE

def tournament(N, fp, pt=0.8):
    xr = np.random.choice(np.arange(N), 2, replace=False)
    r = np.random.rand()
    if r < pt:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[0]
        else:
            p = xr[1]
    else:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[1]
        else:
            p = xr[0]
    return p

def SBX(Pg, N, fp, eta_c=20):
    D = 20
    C = np.empty([0, D])
    for i in range(int(N/2)):
        while 1==1:
            p1 = tournament(N, fp)
            p2 = tournament(N, fp)
            if p1!=p2:
                break
        P1 = Pg[p1]
        P2 = Pg[p2]
        C1 = np.zeros_like(P1)
        C2 = np.zeros_like(P2)
        for j in range(D):
            u = np.random.uniform()
            if u < 0.5:
                beta = (2*u)**(1/(eta_c+1))
            else:
                beta = 1/((2*(1-u))**(1/(eta_c+1)))
            C1[j] = 0.5*((1-beta)*P1[j] + (1+beta)*P2[j])
            C2[j] = 0.5*((1+beta)*P1[j] + (1-beta)*P2[j])
        C = np.row_stack((C, C1, C2))
    return C

def mutation(C, N, Omega, eta_m=20):
    D = 20
    Q = np.empty([0, D])
    for i in range(N):
        p = C[i]
        m = np.zeros_like(p)
        for j in range(D):
            r = np.random.uniform()
            if r < 0.5:
                delta = (2*r)**(1/(eta_m+1))-1
            else:
                delta = 1 - (2*(1-r))**(1/(eta_m+1))
            m[j] = p[j] + delta*(Omega[j,1] - Omega[j,0])
        m = np.clip(m, Omega[:,0], Omega[:,1])
        m[8] = round(m[8])
        Q = np.row_stack((Q, m))
    return Q

def elitism(Pg, fp, NE):
    P = Pg.copy()
    f = fp.copy()
    E = np.empty([0, 20])
    for i in range(NE):
        iB = np.argmin(f)
        E = np.row_stack((E, P[iB]))
        P = np.delete(P, iB, axis=0)
        f = np.delete(f, iB)        
    return E

def updateMetrics(metrics, TWQg, fp):
    iopt = np.argmin(fp)
    metrics['T_opt'] = np.append(metrics['T_opt'], TWQg[iopt,0])
    metrics['W_opt'] = np.append(metrics['W_opt'], TWQg[iopt,1])
    metrics['Q_u_opt'] = np.append(metrics['Q_u_opt'], TWQg[iopt,2])
    metrics['Q_l_opt'] = np.append(metrics['Q_l_opt'], TWQg[iopt,3])
    metrics['T_u_opt'] = np.append(metrics['T_u_opt'], TWQg[iopt,4])
    metrics['T_l_opt'] = np.append(metrics['T_l_opt'], TWQg[iopt,5])
    metrics['FM_u_opt'] = np.append(metrics['FM_u_opt'], TWQg[iopt,6])
    metrics['FM_l_opt'] = np.append(metrics['FM_l_opt'], TWQg[iopt,7])
    
    return metrics

def check_convergence(W_history, threshold=0.001, patience=5):

    if len(W_history) < 2:
        return False
    
    consecutive = 0
    
    # Recorrer de la más reciente hacia atrás
    for i in range(len(W_history) - 1, 0, -1):
        curr = W_history[i]
        prev = W_history[i - 1]
        
        if prev == 0:
            variation = 0
        else:
            variation = abs(curr - prev) / abs(prev)
        
        if variation < threshold:
            consecutive += 1
        else:
            break  # Se rompió la racha, no seguir contando
    
    return consecutive >= patience