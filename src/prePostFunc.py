import numpy as np
import shutil
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from src import geoAirfoil as ga

def readInputs(fileInputs):
    f = open(fileInputs)
    dataInput = json.load(f)    
    X = np.array([dataInput['prop_both']['c_d_r'],
                  dataInput['prop_both']['c_d_m'],
                  dataInput['prop_both']['c_d_t'],
                  dataInput['prop_both']['r_cdm'],
                  dataInput['prop_both']['yt_r'],
                  dataInput['prop_both']['yt_m'],
                  dataInput['prop_both']['yt_t'],
                  dataInput['prop_both']['r_yt'],
                  dataInput['prop_both']['B'],
                  dataInput['prop_both']['d'],
                  dataInput['prop_upper']['alpha_r'],
                  dataInput['prop_upper']['alpha_m'],
                  dataInput['prop_upper']['alpha_t'],
                  dataInput['prop_upper']['r_alpham'],
                  dataInput['prop_upper']['n_min'],
                  dataInput['prop_lower']['alpha_r'],
                  dataInput['prop_lower']['alpha_m'],
                  dataInput['prop_lower']['alpha_t'],
                  dataInput['prop_lower']['r_alpham'],
                  dataInput['prop_lower']['n_min'],
                  ])
    return X, dataInput['flow'], dataInput['optimization'], dataInput['paramBEMT']

def initMetrics():
    metrics = {}
    metrics['T_opt'] = np.array([])
    metrics['W_opt'] = np.array([])
    metrics['Q_u_opt'] = np.array([])
    metrics['Q_l_opt'] = np.array([])
    metrics['T_u_opt'] = np.array([])
    metrics['T_l_opt'] = np.array([])
    metrics['FM_u_opt'] = np.array([])
    metrics['FM_l_opt'] = np.array([])
    return metrics

def printPlots(g, metrics, testDir):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)

    fig = plt.figure(2, figsize=(16*cm, 16*cm))
    ax1 = fig.add_subplot(221)
    ax1.plot(np.arange(g+1), metrics['W_opt'], '-k')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Objective function, $W_{total}$ [W]')
    ax1.grid()
    ax2 = fig.add_subplot(222)
    ax2.plot(np.arange(g+1), metrics['FM_u_opt'], '-b', label='$FM_{upper}$')
    ax2.plot(np.arange(g+1), metrics['FM_l_opt'], '-r', label='$FM_{lower}$')
    ax2.legend(loc='best')
    ax2.set_xlabel('Generations')
    ax2.set_ylabel('$FM$')
    ax2.grid()
    ax3 = fig.add_subplot(223)
    ax3.plot(np.arange(g+1), metrics['T_opt'], '-k', label='$T_{total}$')
    ax3.plot(np.arange(g+1), metrics['T_u_opt'], '-b', label='$T_{upper}$')
    ax3.plot(np.arange(g+1), metrics['T_l_opt'], '-r', label='$T_{lower}$')
    ax3.legend(loc='best')
    ax3.set_xlabel('Generations')
    ax3.set_ylabel('$T$ [N]')
    ax3.grid()
    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(g+1), metrics['Q_u_opt'], '-b', label='$Q_{upper}$')
    ax4.plot(np.arange(g+1), metrics['Q_l_opt'], '-r', label='$Q_{upper}$')
    ax4.legend(loc='best')
    ax4.set_xlabel('Generations')
    ax4.set_ylabel('$Q$ [N m]')
    ax4.grid()
    plt.tight_layout()
    fig.savefig(testDir + '/OpenVINT7_metrics.png')
    plt.show()
    
def saveBladeOpt(fp, bladeg, testDir):
    iopt = np.argmin(fp)
    shutil.copytree(bladeg[iopt], testDir+'/propellerOpt')
    
def blade3D(testDir):    
    os.makedirs(testDir+'/propellerOpt/airfoils_blade_upper', exist_ok=True)
    os.makedirs(testDir+'/propellerOpt/airfoils_blade_lower', exist_ok=True)
    
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)
    
    f = open(testDir+'/propellerOpt/info_prop.json')
    blade = json.load(f)
    
    A = np.load(testDir+'/propellerOpt/A.npy')
    
    fig = plt.figure(66)
    ax = fig.add_subplot(projection='3d')
    R = blade['design_parameters']['d']/2
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    
    xtmin = max(blade['info_sections']['xt'])
    
    for i in range(len(A)):
        X, YU, YL = ga.cstN5(A[i])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*blade['info_sections']['c'][i] - xtmin*blade['info_sections']['c'][i]
        Yscale = Ya*blade['info_sections']['c'][i]
        Xphi = Xscale*np.cos(-blade['info_sections']['phi_up'][i]*np.pi/180) - Yscale*np.sin(-blade['info_sections']['phi_up'][i]*np.pi/180)
        Yphi = Xscale*np.sin(-blade['info_sections']['phi_up'][i]*np.pi/180) + Yscale*np.cos(-blade['info_sections']['phi_up'][i]*np.pi/180)
        Z = np.ones_like(Xscale)*blade['info_sections']['r'][i]
        Points = np.column_stack((Xphi, Yphi))
        Points = np.column_stack((Points, Z))*1000
        ax.plot(Xphi, Yphi, Z, '-b')
        fileXFAirfoil = open(testDir+'/propellerOpt/airfoils_blade_upper/airfoil_'+str(i)+'.dat','w')
        for i in range(Xphi.size):
            Pi = ''
            for k in range(3):
                Pi = Pi + str(np.round(Points[i,k], 6))+' '
            Pi = Pi +'\n'
            fileXFAirfoil.write(Pi)
        fileXFAirfoil.close()
    
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(testDir+'/propellerOpt/fig_blade_upper.png')
    plt.close(fig)
    
    fig = plt.figure(99)
    ax = fig.add_subplot(projection='3d')
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    for i in range(len(A)):
        X, YU, YL = ga.cstN5(A[i])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*blade['info_sections']['c'][i] - xtmin*blade['info_sections']['c'][i]
        Yscale = Ya*blade['info_sections']['c'][i]
        Xphi = Xscale*np.cos(-blade['info_sections']['phi_low'][i]*np.pi/180) - Yscale*np.sin(-blade['info_sections']['phi_low'][i]*np.pi/180)
        Yphi = Xscale*np.sin(-blade['info_sections']['phi_low'][i]*np.pi/180) + Yscale*np.cos(-blade['info_sections']['phi_low'][i]*np.pi/180)
        Z = np.ones_like(Xscale)*blade['info_sections']['r'][i]
        Points = np.column_stack((Xphi, Yphi))
        Points = np.column_stack((Points, Z))*1000
        ax.plot(Xphi, Yphi, Z, '-b')
        fileXFAirfoil = open(testDir+'/propellerOpt/airfoils_blade_lower/airfoil_'+str(i)+'.dat','w')
        for i in range(Xphi.size):
            Pi = ''
            for k in range(3):
                Pi = Pi + str(np.round(Points[i,k], 6))+' '
            Pi = Pi +'\n'
            fileXFAirfoil.write(Pi)
        fileXFAirfoil.close()
    
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(testDir+'/propellerOpt/fig_blade_lower.png')
    plt.close(fig)
    
def blade3DVal(bladeDir):    
    os.mkdir(bladeDir+'/airfoils_blade')
    
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)
    
    f = open(bladeDir+'/info_prop.json')
    blade = json.load(f)
    
    A = np.load(bladeDir+'/A.npy')
    
    fig = plt.figure(66)
    ax = fig.add_subplot(projection='3d')
    R = blade['design_parameters']['d']/2
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    
    for i in range(len(A)):
        X, YU, YL = ga.cstN5(A[i])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*blade['info_sections']['c'][i] - blade['info_sections']['xt'][i]*blade['info_sections']['c'][i]
        Yscale = Ya*blade['info_sections']['c'][i]
        Xphi = Xscale*np.cos(-blade['info_sections']['phi'][i]*np.pi/180) - Yscale*np.sin(-blade['info_sections']['phi'][i]*np.pi/180)
        Yphi = Xscale*np.sin(-blade['info_sections']['phi'][i]*np.pi/180) + Yscale*np.cos(-blade['info_sections']['phi'][i]*np.pi/180)
        Z = np.ones_like(Xscale)*blade['info_sections']['r'][i]
        Points = np.column_stack((Xphi, Yphi))
        Points = np.column_stack((Points, Z))*1000
        ax.plot(Xphi, Yphi, Z, '-b')
        fileXFAirfoil = open(bladeDir+'/airfoils_blade/airfoil_'+str(i)+'.dat','w')
        for i in range(Xphi.size):
            Pi = ''
            for k in range(3):
                Pi = Pi + str(np.round(Points[i,k], 6))+' '
            Pi = Pi +'\n'
            fileXFAirfoil.write(Pi)
        fileXFAirfoil.close()
    
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(bladeDir+'/fig_blade.png')
    plt.close(fig)
    
