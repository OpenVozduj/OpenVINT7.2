import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nn import openvint_vae as av
from nn import openvint_mlp as am
from nn import readerGraphics as rg
import tensorflow as tf
import subprocess
import os
import gc
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# import matplotlib as mpl

def clean_gpu_linux():
    try:
        resultado = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        pid_actual = str(os.getpid())
        for pid in resultado.strip().split('\n'):
            pid = pid.strip()
            if pid and pid != pid_actual:
                subprocess.run(['kill', '-9', pid])
    except Exception:
        pass

def loadNN():
    
    clean_gpu_linux()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        
    tf.keras.backend.clear_session()
    gc.collect()
    
    vae = av.OV_VAE()
    vae.build(input_shape=(256, 256, 2))
    vae.load_weights('nn/vae_5_2.weights.h5')
    
    mlp = am.OV_MLP()
    mlp.load_weights('nn/mlp2.weights.h5')
    return mlp, vae

def normParamCST():
    airfoils = np.load('src/A_arad_family.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(airfoils)
    return normXT, scaler  

def precomputeAirfoilImages(scaler, mlp, vae, A):
    An = scaler.transform(A)
    zPred = mlp.predict(An)
    imgPred = vae.decoder.predict(zPred)
    return imgPred

def getCoeffsFast(imgPred_base, Re_flat, alpha_flat, N_r):    
    results = [rg.searchCoeffswithAlphaRe(alpha_flat[i], Re_flat[i], imgPred_base[i % N_r]) 
               for i in range(len(alpha_flat))]
    cl, cd = zip(*results)
    return np.array(cl), np.array(cd)

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


def TWQCoaxial_Oblique_Vectorized(n_min_u, n_min_l, nB, d, r, c, alpha_target_u, alpha_target_l, imgPred,
                                  Z, rho, nu, V_inf=0.0, alpha_shaft_deg=0.0, CT_guess=0.008, N_psi=36):

    R = d / 2.0
    dr = np.gradient(r)
    dr[0] = r[1] - r[0]
    R_hub = r[0]
    N_r = len(r)
    
    # --- CINEMÁTICA INDEPENDIENTE ---
    n_s_u = n_min_u / 60.0
    n_s_l = n_min_l / 60.0
    omega_u = 2 * np.pi * n_s_u
    omega_l = 2 * np.pi * n_s_l
    
    alpha_shaft = np.radians(alpha_shaft_deg)
    V_axial = V_inf * np.cos(alpha_shaft)
    V_trans = V_inf * np.sin(alpha_shaft)
    
    # --- GRIDS 2D (Azimut vs Radio Físico) ---
    psi_array = np.linspace(0, 2*np.pi, N_psi, endpoint=False)
    psi_grid = psi_array[:, None]       # Forma: (N_psi, 1)
    r_grid = r[None, :]                 # Forma: (1, N_r) -> Se expande a (N_psi, N_r)
    dr_grid = dr[None, :]
    c_grid = c[None, :]
    
    # Coordenadas físicas de la pala en cada azimut (Para intersección de estela)
    X_grid = r_grid * np.cos(psi_grid)  # Forma: (N_psi, N_r)
    Y_grid = r_grid * np.sin(psi_grid)  # Forma: (N_psi, N_r)
    
    # Inicialización de velocidades inducidas
    Cw = 0.9
    Va_u = np.full(N_r, inducedVelocityGuess(V_axial, CT_guess, omega_u, R))
    Va_l_wake = np.full(N_r, Va_u[0] / (Cw**2))   # Inducida en la zona de estela
    Va_l_free = np.full(N_r, inducedVelocityGuess(V_axial, CT_guess, omega_l, R)) # Inducida en flujo libre
    
    xi1, xi2 = 0.8, 0.2
    converged_outer = False
    iter_outer = 0
    
    while not converged_outer and iter_outer < 20:
        iter_outer += 1
        Rc = R * Cw
        v_wake_z = V_axial + np.mean(Va_u)
        t_conv = Z / np.maximum(v_wake_z, 1e-3)
        delta_y_wake = V_trans * t_conv  # Desplazamiento lateral de la estela (Flujo Oblicuo)
        
        converged_inner = False
        iter_inner = 0
        
        while not converged_inner and iter_inner < 50:
            iter_inner += 1
            
            # ==========================================
            # 1. CÁLCULO VECTORIAL ROTOR SUPERIOR
            # ==========================================
            Wa_u_grid = V_axial + Va_u[None, :]  
            Vt_u_grid = omega_u * r_grid + V_trans * np.sin(psi_grid)
            Vres_u_grid = np.sqrt(Wa_u_grid**2 + Vt_u_grid**2)
            phi_inflow_u_grid = np.arctan2(Wa_u_grid, Vt_u_grid)
            
            alpha_u_grid = alpha_target_u[None, :] + np.degrees(phi_inflow_u_grid)
            Re_u_grid = Vres_u_grid * c_grid / nu
            
            alpha_u_flat = alpha_u_grid.flatten()
            Re_u_flat = Re_u_grid.flatten()
            cl_u_flat, cd_u_flat = getCoeffsFast(imgPred, Re_u_flat, alpha_u_flat, N_r)
            
            cl_u_grid = cl_u_flat.reshape(N_psi, N_r)
            cd_u_grid = cd_u_flat.reshape(N_psi, N_r)
            CN_u_grid = cl_u_grid * np.cos(phi_inflow_u_grid) - cd_u_grid * np.sin(phi_inflow_u_grid)
            
            sin_phi_u_grid = np.maximum(np.sin(phi_inflow_u_grid), 1e-4)
            f_tip_u_grid = (nB / 2) * (R - r_grid) / (r_grid * sin_phi_u_grid)
            Ftip_u_grid = np.maximum((2 / np.pi) * np.arccos(np.exp(-f_tip_u_grid)), 0.001)
            
            dN_u_psi_grid = 0.5 * rho * Vres_u_grid**2 * c_grid * CN_u_grid * Ftip_u_grid * nB * dr_grid
            dN_u_avg = np.mean(dN_u_psi_grid, axis=0)
            
            # ==========================================
            # 2. CÁLCULO VECTORIAL ROTOR INFERIOR (FÍSICA REAL)
            # ==========================================
            # A. Intersección de Estela Oblicua (Máscara 2D)
            dist_to_wake = np.sqrt(X_grid**2 + (Y_grid - delta_y_wake)**2)
            in_wake_mask = dist_to_wake <= Rc  # Forma: (N_psi, N_r)
            
            # B. Campo de Velocidad Axial Combinado
            # Donde hay estela, usamos la inducción del superior. Donde no, usamos la inducción libre local.
            Va_l_combined_grid = np.where(in_wake_mask, Va_l_wake[None, :], Va_l_free[None, :])
            Wa_l_grid = V_axial + Va_l_combined_grid
            
            Vt_l_grid = omega_l * r_grid + V_trans * np.sin(psi_grid)
            Vres_l_grid = np.sqrt(Wa_l_grid**2 + Vt_l_grid**2)
            phi_inflow_l_grid = np.arctan2(Wa_l_grid, Vt_l_grid)
            
            alpha_l_grid = alpha_target_l[None, :] + np.degrees(phi_inflow_l_grid)
            Re_l_grid = Vres_l_grid * c_grid / nu
            
            alpha_l_flat = alpha_l_grid.flatten()
            Re_l_flat = Re_l_grid.flatten()
            cl_l_flat, cd_l_flat = getCoeffsFast(imgPred, Re_l_flat, alpha_l_flat, N_r)
            
            cl_l_grid = cl_l_flat.reshape(N_psi, N_r)
            cd_l_grid = cd_l_flat.reshape(N_psi, N_r)
            CN_l_grid = cl_l_grid * np.cos(phi_inflow_l_grid) - cd_l_grid * np.sin(phi_inflow_l_grid)
            
            sin_phi_l_grid = np.maximum(np.sin(phi_inflow_l_grid), 1e-4)
            f_tip_l_grid = (nB / 2) * (R - r_grid) / (r_grid * sin_phi_l_grid)
            Ftip_l_grid = np.maximum((2 / np.pi) * np.arccos(np.exp(-f_tip_l_grid)), 0.001)
            
            dN_l_psi_grid = 0.5 * rho * Vres_l_grid**2 * c_grid * CN_l_grid * Ftip_l_grid * nB * dr_grid
            
            # C. Separación de Fuerzas para la Teoría de Momento Dual
            dN_l_wake_avg = np.mean(np.where(in_wake_mask, dN_l_psi_grid, 0.0), axis=0)
            dN_l_free_avg = np.mean(np.where(~in_wake_mask, dN_l_psi_grid, 0.0), axis=0)
            dN_l_avg_total = dN_l_wake_avg + dN_l_free_avg
            
            # ==========================================
            # 3. TEORÍA DE MOMENTO LOCAL (DUAL)
            # ==========================================
            dA = 2 * np.pi * r * dr
            
            # Rotor Superior
            term_u = (V_axial**2 / 4.0) + (dN_u_avg / (2.0 * rho * dA + 1e-9))
            vi_u_upd = - (V_axial / 2.0) + np.sqrt(np.maximum(term_u, 0.0))
            
            # Rotor Inferior: Zona de Estela (Arrastra la inducción del superior)
            vi_l_wake_upd = vi_u_upd / (Cw**2)
            
            # Rotor Inferior: Zona de Flujo Libre (Teoría de momento propia, como rotor aislado)
            term_l_free = (V_axial**2 / 4.0) + (dN_l_free_avg / (2.0 * rho * dA + 1e-9))
            vi_l_free_upd = - (V_axial / 2.0) + np.sqrt(np.maximum(term_l_free, 0.0))
            
            # Actualización y Relajación
            Va_u_new = xi1 * Va_u + xi2 * np.clip(vi_u_upd, 0.01, omega_u * R * 0.5)
            Va_l_wake_new = xi1 * Va_l_wake + xi2 * np.clip(vi_l_wake_upd, 0.01, omega_l * R * 0.5)
            Va_l_free_new = xi1 * Va_l_free + xi2 * np.clip(vi_l_free_upd, 0.01, omega_l * R * 0.5)
            
            err_u = np.max(np.abs(Va_u_new - Va_u)) / (np.max(Va_u) + 1e-6)
            err_l_wake = np.max(np.abs(Va_l_wake_new - Va_l_wake)) / (np.max(Va_l_wake) + 1e-6)
            err_l_free = np.max(np.abs(Va_l_free_new - Va_l_free)) / (np.max(Va_l_free) + 1e-6)
            
            if max(err_u, err_l_wake, err_l_free) < 0.005: converged_inner = True
            Va_u, Va_l_wake, Va_l_free = Va_u_new, Va_l_wake_new, Va_l_free_new
            
        # --- Landgrebe y Contracción de Estela ---
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
    # INTEGRACIÓN FINAL Y MÉTRICAS
    # ==========================================
    Thrust_u = np.sum(dN_u_avg)
    Thrust_l = np.sum(dN_l_avg_total)
    Thrust_total = Thrust_u + Thrust_l
    
    # Torques
    CT_u_grid = cl_u_grid * np.sin(phi_inflow_u_grid) + cd_u_grid * np.cos(phi_inflow_u_grid)
    dQ_u_psi_grid = 0.5 * rho * Vres_u_grid**2 * c_grid * CT_u_grid * Ftip_u_grid * nB * r_grid * dr_grid
    Q_u = np.sum(np.mean(dQ_u_psi_grid, axis=0))
    
    CT_l_grid = cl_l_grid * np.sin(phi_inflow_l_grid) + cd_l_grid * np.cos(phi_inflow_l_grid)
    dQ_l_psi_grid = 0.5 * rho * Vres_l_grid**2 * c_grid * CT_l_grid * Ftip_l_grid * nB * r_grid * dr_grid
    Q_l = np.sum(np.mean(dQ_l_psi_grid, axis=0))
    
    # Potencias
    Power_u = Q_u * omega_u
    Power_l = Q_l * omega_l
    Power_total = Power_u + Power_l
    P_net_total = Power_total - Thrust_total * V_axial
    
    # Figuras de Mérito
    FM_u, _, _ = calculateClimbMetrics(Thrust_u, Power_u, V_axial, rho, R)
    FM_l, _, _ = calculateClimbMetrics(Thrust_l, Power_l, V_axial, rho, R)
    
    # Cálculo de phi promedio para salida (¡Sin ceros artificiales!)
    phi_u_avg_out = np.mean(alpha_target_u[None, :] + np.degrees(phi_inflow_u_grid), axis=0)
    phi_l_avg_out = np.mean(alpha_target_l[None, :] + np.degrees(phi_inflow_l_grid), axis=0)
    
    # Métricas adimensionales y fracción de estela
    alpha_p = Thrust_total / (rho * n_s_u**2 * d**4)
    beta_p = Power_total / (rho * n_s_u**3 * d**5)
    Torque_eq = Power_total / (2 * np.pi * n_s_u) 
    f_wake_approx = np.clip(1.0 - (abs(delta_y_wake) / (R + Rc)), 0.0, 1.0) if (R + Rc) > 0 else 1.0
    
    return (Thrust_total, Power_total, alpha_p, beta_p, Torque_eq, Cw, f_wake_approx,
            Thrust_u, Thrust_l, Power_u, Power_l, Q_u, Q_l,
            FM_u, FM_l, P_net_total, phi_u_avg_out, phi_l_avg_out)
