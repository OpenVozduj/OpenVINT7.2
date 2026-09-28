import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline, interpn
from scipy.optimize import curve_fit

def setup_aero_surrogate(n_components=5):
    
    alpha_grid = np.arange(-5, 11, 1) 
    Re_grid = np.array([50000, 100000, 200000, 300000, 400000, 500000, 600000])
    yt_array = np.linspace(5, 16, 100)
    CY_tensor = np.load('src/CY_tensor.npy')
    CX_tensor = np.load('src/CX_tensor.npy')
    
    yt_array = np.array(yt_array)
    alpha_grid = np.array(alpha_grid)
    Re_grid = np.array(Re_grid)
    
    # Verificación de seguridad: si por accidente se cargó al revés, lo transponemos
    if CY_tensor.shape[1] == len(alpha_grid) and CY_tensor.shape[2] == len(Re_grid):
        CY_tensor = np.transpose(CY_tensor, (0, 2, 1))
        CX_tensor = np.transpose(CX_tensor, (0, 2, 1))
        
    N_Re = len(Re_grid)
    N_alpha = len(alpha_grid)
    
    # 1. PCA para CY
    CY_flat = CY_tensor.reshape(len(yt_array), -1)
    pca_cy = PCA(n_components=n_components)
    scores_cy = pca_cy.fit_transform(CY_flat)
    
    splines_cy = [CubicSpline(yt_array, scores_cy[:, i]) for i in range(n_components)]
    
    # 2. Ajuste de Polara para CX
    def polar_drag(Cy, Cx0, k): return Cx0 + k * Cy**2
    Cx0_array = np.zeros(len(yt_array))
    k_array = np.zeros(len(yt_array))
    
    for i in range(len(yt_array)):
        cy_pts = CY_tensor[i].flatten()
        cx_pts = CX_tensor[i].flatten()
        mask = (cy_pts > -0.5) & (cy_pts < 1.5)
        popt, _ = curve_fit(polar_drag, cy_pts[mask], cx_pts[mask], p0=[0.01, 0.05])
        Cx0_array[i], k_array[i] = popt[0], popt[1]
        
    spline_Cx0 = CubicSpline(yt_array, Cx0_array)
    spline_k = CubicSpline(yt_array, k_array)
    
    model_state = {
        'pca_cy': pca_cy,
        'splines_cy': splines_cy,
        'spline_Cx0': spline_Cx0,
        'spline_k': spline_k,
        'alpha_grid': alpha_grid,
        'Re_grid': Re_grid,
        'N_alpha': N_alpha,
        'N_Re': N_Re
    }
    
    print(f"✅ Modelo sustituto configurado. Tensores validados con forma (N_perfiles, {N_Re}, {N_alpha}).")
    return model_state

def reconstruct_aero_fields(model_state, yt_batch):
    
    yt_batch = np.atleast_1d(yt_batch*100)
    N = len(yt_batch)
    
    scores_new = np.column_stack([spline(yt_batch) for spline in model_state['splines_cy']])
    
    CY_flat_new = scores_new @ model_state['pca_cy'].components_ + model_state['pca_cy'].mean_
    CY_batch = CY_flat_new.reshape(N, model_state['N_Re'], model_state['N_alpha'])
    
    Cx0_new = model_state['spline_Cx0'](yt_batch)[:, None, None]
    k_new = model_state['spline_k'](yt_batch)[:, None, None]
    CX_batch = Cx0_new + k_new * (CY_batch ** 2)
    
    return CY_batch, CX_batch

def read_specific_aero_values(model_state, CY_batch, CX_batch, alpha_targets, re_targets):
    for i in range(len(re_targets)):
        if re_targets[i] < 50000:
            re_targets[i] = 50000
        if re_targets[i] > 600000:
            re_targets[i] = 600000
        
    alpha_targets = np.atleast_1d(alpha_targets)
    re_targets = np.atleast_1d(re_targets)
    N = len(alpha_targets)
    points = (model_state['Re_grid'], model_state['alpha_grid'])
    
    Cl_vals = np.zeros(N)
    Cd_vals = np.zeros(N)
    
    for i in range(N):
        pt = (re_targets[i], alpha_targets[i])
        
        Cl_vals[i] = interpn(points, CY_batch[i], pt, method='linear', 
                             bounds_error=False, fill_value=None)[0]
        Cd_vals[i] = interpn(points, CX_batch[i], pt, method='linear', 
                             bounds_error=False, fill_value=None)[0]
        
    return Cl_vals, Cd_vals

def getCoeffsFast(model_state, CY_batch, CX_batch, Re_flat, alpha_flat, N_r):    
    N_psi = int(len(Re_flat) / N_r)
    cl = np.array([])
    cd = np.array([])
    for i in range(N_psi):
        alpha_psi_i = alpha_flat[i*N_r:(i+1)*N_r]
        Re_psi_i = Re_flat[i*N_r:(i+1)*N_r]
        cl_psi, cd_psi = read_specific_aero_values(model_state, CY_batch, CX_batch,
                                                   alpha_psi_i, Re_psi_i)
        cl = np.append(cl, cl_psi)
        cd = np.append(cd, cd_psi)
    return cl, cd

def calculateClimbMetrics(Thrust, Power, V_axial, rho, R):
    A = np.pi * R**2
    if Thrust > 0 and Power > 0:
        v_h = np.sqrt(Thrust / (2.0 * rho * A))
        P_net = max(Power - Thrust * V_axial, 0.0)
        P_ideal_hover = Thrust * v_h
        FM_hover_eq = P_ideal_hover / (P_net + P_ideal_hover)
        return FM_hover_eq, P_ideal_hover, P_net
    else:
        return 0.0, 0.0, 0.0

def inducedVelocityGuess(V_axial, CT_guess, omega, R):
    lambda_c = V_axial / (omega * R)
    lambda_i = -0.5 * lambda_c + np.sqrt(0.25 * lambda_c**2 + 0.5 * CT_guess)
    return lambda_i * omega * R


def TWQCoaxial_Oblique_Vectorized(n_min_u, n_min_l, nB, d, r, c, alpha_target_u, alpha_target_l, CY_batch,
                                  CX_batch, model, Z, rho, nu, V_inf=0.0, alpha_shaft_deg=0.0, 
                                  CT_guess=0.008, N_psi=36):

    R = d / 2.0
    dr = np.gradient(r)
    dr[0] = r[1] - r[0]
    R_hub = r[0]
    N_r = len(r)
    
    # 
    n_s_u = n_min_u / 60.0
    n_s_l = n_min_l / 60.0
    omega_u = 2 * np.pi * n_s_u
    omega_l = 2 * np.pi * n_s_l
    
    alpha_shaft = np.radians(alpha_shaft_deg)
    V_axial = V_inf * np.cos(alpha_shaft)
    V_trans = V_inf * np.sin(alpha_shaft)
    
    # 
    psi_array = np.linspace(0, 2*np.pi, N_psi, endpoint=False)
    psi_grid = psi_array[:, None]       
    r_grid = r[None, :]              
    dr_grid = dr[None, :]
    c_grid = c[None, :]
    
    # 
    X_grid = r_grid * np.cos(psi_grid)  
    Y_grid = r_grid * np.sin(psi_grid)  
    
    # 
    Cw = 0.9
    Va_u = np.full(N_r, inducedVelocityGuess(V_axial, CT_guess, omega_u, R))
    Va_l_wake = np.full(N_r, Va_u[0] / (Cw**2))   
    Va_l_free = np.full(N_r, inducedVelocityGuess(V_axial, CT_guess, omega_l, R)) 
    
    xi1, xi2 = 0.8, 0.2
    converged_outer = False
    iter_outer = 0
    
    while not converged_outer and iter_outer < 20:
        iter_outer += 1
        Rc = R * Cw
        v_wake_z = V_axial + np.mean(Va_u)
        t_conv = Z / np.maximum(v_wake_z, 1e-3)
        delta_y_wake = V_trans * t_conv  
        
        converged_inner = False
        iter_inner = 0
        
        while not converged_inner and iter_inner < 50:
            iter_inner += 1
            
            # ==========================================
            Wa_u_grid = V_axial + Va_u[None, :]  
            Vt_u_grid = omega_u * r_grid + V_trans * np.sin(psi_grid)
            Vres_u_grid = np.sqrt(Wa_u_grid**2 + Vt_u_grid**2)
            phi_inflow_u_grid = np.arctan2(Wa_u_grid, Vt_u_grid)
            
            # 
            alpha_u_grid = np.tile(alpha_target_u, (N_psi, 1)) 
            Re_u_grid = Vres_u_grid * c_grid / nu
            
            alpha_u_flat = alpha_u_grid.flatten()
            Re_u_flat = Re_u_grid.flatten()
            cl_u_flat, cd_u_flat = getCoeffsFast(model, CY_batch, CX_batch, Re_u_flat,
                                                 alpha_u_flat, N_r)
            
            cl_u_grid = cl_u_flat.reshape(N_psi, N_r)
            cd_u_grid = cd_u_flat.reshape(N_psi, N_r)
            CN_u_grid = cl_u_grid * np.cos(phi_inflow_u_grid) - cd_u_grid * np.sin(phi_inflow_u_grid)
            
            sin_phi_u_grid = np.maximum(np.sin(phi_inflow_u_grid), 1e-4)
            f_tip_u_grid = (nB / 2) * (R - r_grid) / (r_grid * sin_phi_u_grid)
            Ftip_u_grid = np.maximum((2 / np.pi) * np.arccos(np.exp(-f_tip_u_grid)), 0.001)
            
            dN_u_psi_grid = 0.5 * rho * Vres_u_grid**2 * c_grid * CN_u_grid * Ftip_u_grid * nB * dr_grid
            dN_u_avg = np.mean(dN_u_psi_grid, axis=0)
            
            # 
            phi_geo_u_grid = alpha_u_grid + np.degrees(phi_inflow_u_grid)
            
            # ==========================================
            
            # 
            dist_to_wake = np.sqrt(X_grid**2 + (Y_grid - delta_y_wake)**2)
            in_wake_mask = dist_to_wake <= Rc  
            
            # 
            Va_l_combined_grid = np.where(in_wake_mask, Va_l_wake[None, :], Va_l_free[None, :])
            Wa_l_grid = V_axial + Va_l_combined_grid
            
            Vt_l_grid = omega_l * r_grid + V_trans * np.sin(psi_grid)
            Vres_l_grid = np.sqrt(Wa_l_grid**2 + Vt_l_grid**2)
            phi_inflow_l_grid = np.arctan2(Wa_l_grid, Vt_l_grid)
            
            # 
            alpha_l_grid = np.tile(alpha_target_l, (N_psi, 1))
            Re_l_grid = Vres_l_grid * c_grid / nu
            
            alpha_l_flat = alpha_l_grid.flatten()
            Re_l_flat = Re_l_grid.flatten()
            cl_l_flat, cd_l_flat = getCoeffsFast(model, CY_batch, CX_batch, Re_l_flat,
                                                 alpha_l_flat, N_r)
            cl_l_grid = cl_l_flat.reshape(N_psi, N_r)
            cd_l_grid = cd_l_flat.reshape(N_psi, N_r)
            CN_l_grid = cl_l_grid * np.cos(phi_inflow_l_grid) - cd_l_grid * np.sin(phi_inflow_l_grid)
            
            sin_phi_l_grid = np.maximum(np.sin(phi_inflow_l_grid), 1e-4)
            f_tip_l_grid = (nB / 2) * (R - r_grid) / (r_grid * sin_phi_l_grid)
            Ftip_l_grid = np.maximum((2 / np.pi) * np.arccos(np.exp(-f_tip_l_grid)), 0.001)
            
            dN_l_psi_grid = 0.5 * rho * Vres_l_grid**2 * c_grid * CN_l_grid * Ftip_l_grid * nB * dr_grid
            
            # 
            dN_l_wake_avg = np.mean(np.where(in_wake_mask, dN_l_psi_grid, 0.0), axis=0)
            dN_l_free_avg = np.mean(np.where(~in_wake_mask, dN_l_psi_grid, 0.0), axis=0)
            dN_l_avg_total = dN_l_wake_avg + dN_l_free_avg
            
            phi_geo_l_grid = alpha_l_grid + np.degrees(phi_inflow_l_grid)
            
            # ==========================================
            dA = 2 * np.pi * r * dr
            
            #
            term_u = (V_axial**2 / 4.0) + (dN_u_avg / (2.0 * rho * dA + 1e-9))
            vi_u_upd = - (V_axial / 2.0) + np.sqrt(np.maximum(term_u, 0.0))
            
            # 
            vi_l_wake_upd = vi_u_upd / (Cw**2)
            
            #
            term_l_free = (V_axial**2 / 4.0) + (dN_l_free_avg / (2.0 * rho * dA + 1e-9))
            vi_l_free_upd = - (V_axial / 2.0) + np.sqrt(np.maximum(term_l_free, 0.0))
            
            # 
            Va_u_new = xi1 * Va_u + xi2 * np.clip(vi_u_upd, 0.01, omega_u * R * 0.5)
            Va_l_wake_new = xi1 * Va_l_wake + xi2 * np.clip(vi_l_wake_upd, 0.01, omega_l * R * 0.5)
            Va_l_free_new = xi1 * Va_l_free + xi2 * np.clip(vi_l_free_upd, 0.01, omega_l * R * 0.5)
            
            err_u = np.max(np.abs(Va_u_new - Va_u)) / (np.max(Va_u) + 1e-6)
            err_l_wake = np.max(np.abs(Va_l_wake_new - Va_l_wake)) / (np.max(Va_l_wake) + 1e-6)
            err_l_free = np.max(np.abs(Va_l_free_new - Va_l_free)) / (np.max(Va_l_free) + 1e-6)
            
            if max(err_u, err_l_wake, err_l_free) < 0.005: converged_inner = True
            Va_u, Va_l_wake, Va_l_free = Va_u_new, Va_l_wake_new, Va_l_free_new
            
        # 
        Thrust_u_avg = np.sum(dN_u_avg)
        Ct_u = np.maximum(Thrust_u_avg / (rho * omega_u**2 * np.pi * R**4), 1e-6)
        sigma = nB * np.mean(c) / (np.pi * R)
        
        phi_u_avg = np.mean(alpha_target_u[None, :] + np.degrees(phi_inflow_u_grid), axis=0)
        theta_tw = phi_u_avg[-1] - phi_u_avg[0]
        
        K1 = 0.25 * (Ct_u / sigma + 0.001 * theta_tw)
        K2 = (1 + 0.01 * theta_tw) * np.sqrt(Ct_u)
        Lambda = 0.145 + 27 * Ct_u
        K3 = 0.78
        psi_1 = 2 * np.pi / nB
        conv_factor = 1.0 + (V_axial / np.maximum(np.mean(Va_u), 1e-3))
        psi_w = (Z / (R * K1)) / conv_factor
        if psi_w > psi_1: psi_w = (Z/R - K1 * psi_1) / K2 + psi_1
        psi_w = np.maximum(psi_w, 1e-6)
        
        Cw_upd = K3 + (1 - K3) * np.exp(-Lambda * psi_w)
        Cw_new = xi1 * Cw + xi2 * np.clip(Cw_upd, 0.50, 0.99)
        if abs(Cw_new - Cw) < 0.005: converged_outer = True
        Cw = np.clip(Cw_new, 0.50, 0.99)
        
    # ==========================================

    Thrust_u = np.sum(dN_u_avg)
    Thrust_l = np.sum(dN_l_avg_total)
    Thrust_total = Thrust_u + Thrust_l
    
    #
    CT_u_grid = cl_u_grid * np.sin(phi_inflow_u_grid) + cd_u_grid * np.cos(phi_inflow_u_grid)
    dQ_u_psi_grid = 0.5 * rho * Vres_u_grid**2 * c_grid * CT_u_grid * Ftip_u_grid * nB * r_grid * dr_grid
    Q_u = np.sum(np.mean(dQ_u_psi_grid, axis=0))
    
    CT_l_grid = cl_l_grid * np.sin(phi_inflow_l_grid) + cd_l_grid * np.cos(phi_inflow_l_grid)
    dQ_l_psi_grid = 0.5 * rho * Vres_l_grid**2 * c_grid * CT_l_grid * Ftip_l_grid * nB * r_grid * dr_grid
    Q_l = np.sum(np.mean(dQ_l_psi_grid, axis=0))
    
    # 
    Power_u = Q_u * omega_u
    Power_l = Q_l * omega_l
    Power_total = Power_u + Power_l
    P_net_total = Power_total - Thrust_total * V_axial
    
    # 
    FM_u, _, _ = calculateClimbMetrics(Thrust_u, Power_u, V_axial, rho, R)
    FM_l, _, _ = calculateClimbMetrics(Thrust_l, Power_l, V_axial, rho, R)
    
    # 
    phi_u_avg_out = np.mean(phi_geo_u_grid, axis=0)
    phi_l_avg_out = np.mean(phi_geo_l_grid, axis=0)
    
    # 
    alpha_p = Thrust_total / (rho * n_s_u**2 * d**4)
    beta_p = Power_total / (rho * n_s_u**3 * d**5)
    Torque_eq = Power_total / (2 * np.pi * n_s_u) 
    f_wake_approx = np.clip(1.0 - (abs(delta_y_wake) / (R + Rc)), 0.0, 1.0) if (R + Rc) > 0 else 1.0
    
    return (Thrust_total, Power_total, alpha_p, beta_p, Torque_eq, Cw, f_wake_approx,
            Thrust_u, Thrust_l, Power_u, Power_l, Q_u, Q_l,
            FM_u, FM_l, P_net_total, phi_u_avg_out, phi_l_avg_out)
