import numpy as np
import math

def cstN5(A, N1=0.5, N2=1.0, n=71):
    X = 1 - np.cos((np.arange(n) * np.pi) / (2 * (n - 1)))
    C = (X**N1) * ((1 - X)**N2)
    AU = A[:6]
    AL = A[6:]
    N = 5 
    
    SU = np.zeros_like(X)
    SL = np.zeros_like(X)
    for i in range(N + 1):
        bernstein = math.comb(N, i) * (X**i) * ((1 - X)**(N - i))
        SU += AU[i] * bernstein
        SL += AL[i] * bernstein
        
    return X, C * SU, C * SL

def cst_ST_SC(A, N1=0.5, N2=1.0, n=71):
    X = 1 - np.cos((np.arange(n) * np.pi) / (2 * (n - 1)))
    C = (X**N1) * ((1 - X)**N2)
    AU = A[:6]
    AL = A[6:]
    N = 5
    
    ST = np.zeros_like(X)
    SC = np.zeros_like(X)
    for i in range(N + 1):
        bernstein = math.comb(N, i) * (X**i) * ((1 - X)**(N - i))
        ST += 0.5 * (AU[i] - AL[i]) * bernstein
        SC += 0.5 * (AU[i] + AL[i]) * bernstein
        
    YT = 2 * C * ST
    YC = C * SC
    return X, YT, YC

def get_cst_for_yt(yt_target, yt_ref, A_ref):
    num_coeffs = A_ref.shape[1]
    A_target = np.zeros(num_coeffs)
    
    for i in range(num_coeffs):
        coeffs_poly = np.polyfit(yt_ref, A_ref[:, i], deg=2)
        
        A_target[i] = np.polyval(coeffs_poly, yt_target)
        
    return A_target

def creatorAirfoil(yt):
    yt_new = yt * 100
    yt_ref = np.array([6.0, 10.0, 13.0]) 
    A_arad6 = np.load('src/arad6_cst5.npy')
    A_arad10 = np.load('src/arad10_cst5.npy')
    A_arad13 = np.load('src/arad13_cst5.npy')

    A_arads = np.vstack((A_arad6, A_arad10))
    A_arads = np.vstack((A_arads, A_arad13))

    A_upper_ref = A_arads[:,:6]
    A_lower_ref = A_arads[:,6:]
    A_upper_new = get_cst_for_yt(yt_new, yt_ref, A_upper_ref)
    A_lower_new = get_cst_for_yt(yt_new, yt_ref, A_lower_ref)
    
    A = np.append(A_upper_new, A_lower_new)
    
    return A

