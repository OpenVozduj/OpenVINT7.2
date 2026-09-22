import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import InterpolatedUnivariateSpline
from math import factorial
from src import geoAirfoil as ga

def BezierN(p, n):
    nc = 20
    t = np.linspace(0, 1, nc)
    B = 0
    for i in range(n+1):
        B += p[i]*t**i*(1-t)**(n-i)*(factorial(n)/(factorial(i)*factorial(n-i)))
    return B

def chordCurve(Pi, r, nameBlade):
    d = Pi[9]
    R = d/2    
    pR_r = np.array([r[0], 0.5*(Pi[3]*R+r[0]), Pi[3]*R])
    pc_r = np.array([Pi[0]*d, Pi[1]*d, Pi[1]*d])
    pR_t = np.array([Pi[3]*R, (0.485+0.5*Pi[3])*R, r[-1]])
    pc_t = np.array([Pi[1]*d, Pi[1]*d, Pi[2]*d])
    BR_r = BezierN(pR_r, 2)
    Bc_r = BezierN(pc_r, 2)
    BR_t = BezierN(pR_t, 2)
    Bc_t = BezierN(pc_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Bc = np.append(Bc_r[:-1], Bc_t)
    FC = InterpolatedUnivariateSpline(BR, Bc, k=4)
    c = FC(r)
    return c

def alphaCurve(Pi, r, R, nameBlade):
    pR_r_up = np.array([r[0], 0.5*(Pi[13]*R+r[0]), Pi[13]*R])
    palpha_r_up = np.array([Pi[10], Pi[11], Pi[11]])
    pR_t_up = np.array([Pi[13]*R, (0.485+0.5*Pi[13])*R, r[-1]])
    palpha_t_up = np.array([Pi[11], Pi[11], Pi[12]])
    BR_r_up = BezierN(pR_r_up, 2)
    Balpha_r_up = BezierN(palpha_r_up, 2)
    BR_t_up = BezierN(pR_t_up, 2)
    Balpha_t_up = BezierN(palpha_t_up, 2)
    BR_up = np.append(BR_r_up[:-1], BR_t_up)
    Balpha_up = np.append(Balpha_r_up[:-1], Balpha_t_up)
    Falpha_up = InterpolatedUnivariateSpline(BR_up, Balpha_up, k=4)
    alpha_up = Falpha_up(r)
    
    pR_r_low = np.array([r[0], 0.5*(Pi[18]*R+r[0]), Pi[18]*R])
    palpha_r_low = np.array([Pi[15], Pi[16], Pi[16]])
    pR_t_low = np.array([Pi[18]*R, (0.485+0.5*Pi[18])*R, r[-1]])
    palpha_t_low = np.array([Pi[16], Pi[16], Pi[17]])
    BR_r_low = BezierN(pR_r_low, 2)
    Balpha_r_low = BezierN(palpha_r_low, 2)
    BR_t_low = BezierN(pR_t_low, 2)
    Balpha_t_low = BezierN(palpha_t_low, 2)
    BR_low = np.append(BR_r_low[:-1], BR_t_low)
    Balpha_low = np.append(Balpha_r_low[:-1], Balpha_t_low)
    Falpha_low = InterpolatedUnivariateSpline(BR_low, Balpha_low, k=4)
    alpha_low = Falpha_low(r)
    
    return alpha_up, alpha_low

def airfoilCurves(Pi, r, R, nameBlade):    
    pR_r = np.array([r[0], 0.5*(Pi[7]*R+r[0]), Pi[7]*R])
    pyt_r = np.array([Pi[4], Pi[5], Pi[5]])
    pR_t = np.array([Pi[7]*R, (0.485+0.5*Pi[7])*R, r[-1]])
    pyt_t = np.array([Pi[5], Pi[5], Pi[6]])
    BR_r = BezierN(pR_r, 2)
    Byt_r = BezierN(pyt_r, 2)
    BR_t = BezierN(pR_t, 2)
    Byt_t = BezierN(pyt_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Byt = np.append(Byt_r[:-1], Byt_t)
    Fyt = InterpolatedUnivariateSpline(BR, Byt, k=4)
    ytMax = Fyt(r)

    return ytMax

def paramBlade(Pi, nameBlade, rR_min, rR_max, sect_body, sect_tip):
    r_R1 = np.linspace(rR_min, 0.75, sect_body)
    r_R2 = np.linspace(0.776, rR_max, sect_tip)
    r_R = np.append(r_R1, r_R2)
    R = Pi[9]/2
    r = r_R*R
    c = chordCurve(Pi, r, nameBlade)
    ytMax = airfoilCurves(Pi, r, R, nameBlade)
    alpha_up, alpha_low = alphaCurve(Pi, r, R, nameBlade)
        
    A = np.empty([0,12])
    xt = np.array([])
    yt = np.array([])
    xc = np.array([])
    yc = np.array([])
    for s in range(len(r)):
        As = ga.creatorAirfoil(ytMax[s])
        A = np.row_stack((A, As))
        X, YU, YL = ga.cstN5(As)
        _, YT, YC = ga.cst_ST_SC(As)
        xt = np.append(xt, X[np.argmax(YT)])
        yt = np.append(yt, max(YT))
        xc = np.append(xc, X[np.argmax(YC)])
        yc = np.append(yc, max(YC))
    
    np.save(nameBlade+'/A.npy', A)
    
    dS = np.column_stack((A, r_R))
    dS = np.column_stack((dS, c))
    dS = np.column_stack((dS, alpha_up))
    dS = np.column_stack((dS, alpha_low))
    dS = np.column_stack((dS, r))
    dS = np.column_stack((dS, xt))
    dS = np.column_stack((dS, yt))
    dS = np.column_stack((dS, xc))
    dS = np.column_stack((dS, yc))
    
    return dS

def drawBlade(nameBlade, c, xt, r, phi, R, r_R):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)

    le = -xt*c
    te = c-xt*c

    xle = le*np.cos(-phi*np.pi/180)
    yle = le*np.sin(-phi*np.pi/180)
    xte = te*np.cos(-phi*np.pi/180)
    yte = te*np.sin(-phi*np.pi/180)

    fig = plt.figure(16)
    ax = fig.add_subplot(projection='3d')
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    for ns in range(len(r)):
        ax.plot(np.array([xle[ns], xte[ns]]), np.array([yle[ns], yte[ns]]), np.ones(2)*r[ns], '-g')
    ax.plot(xle, yle, r, '-b')
    ax.plot(xte, yte, r, '-r')
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(nameBlade+'/fig_blade.png')
    plt.close(fig)
    # plt.show()
    
    fig = plt.figure(17, figsize=(12*cm, 4*cm))
    ax = plt.subplot(111)
    ax.plot(r_R, phi, '-b')
    ax.set_xlabel('r/R')
    ax.set_ylabel('$\phi$ [°]')
    ax.grid()
    plt.tight_layout()
    fig.savefig(nameBlade+'/fig_phi_Function.png')
    plt.close(fig)
    # plt.show()