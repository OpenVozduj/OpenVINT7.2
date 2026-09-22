#%% Libraries
import numpy as np
import os
from src import prePostFunc as pp
from src import operGenetics as og
from src import aeroPropeller as ap
import time

#%% Load neural network
mlp, vae = ap.loadNN()
_, scaler = ap.normParamCST()

#%% Preprocess
t0 = time.time()
fileInputs = 'inputs.json'
Omega, flow, opt, paramBEMT = pp.readInputs(fileInputs)
rootDir = os.getcwd()
Metrics = pp.initMetrics()

#%% Initial population
print('Start optimization \n')
print('Genration 0 \n')
os.makedirs('tests/' + opt['testDir'], exist_ok=True)
fileGen = 'tests/' + opt['testDir'] + '/G0'
os.makedirs(fileGen, exist_ok=True)

Pg, bladeg = og.initialPopulation(fileGen, opt['N'], Omega)
TWQg = og.objectiveFunction(opt['N'], Pg, bladeg, scaler, mlp, vae, flow, paramBEMT)

#%%
fn, v, d = og.distanceX(TWQg[:,1], TWQg[:,0], opt['Tmin'], TWQg[:,2], TWQg[:,3], opt['Qtol'])
p = og.twoPenalties(fn, v)
fp = p+d
Metrics = og.updateMetrics(Metrics, TWQg, fp)
NEF = opt['N']

#%% Generations
for g in range(1, opt['G']):
    print('Genration '+str(g)+'\n')
    fileGen = 'tests/' + opt['testDir'] + '/G' + str(g)
    os.makedirs(fileGen, exist_ok=True)
    Ng, NE = og.newSizePop(opt['N'], opt['G'], g)
    E = og.elitism(Pg, fp, NE)
    C = og.SBX(Pg, Ng-NE, fp)
    Q = og.mutation(C, Ng-NE, Omega)
    
    Pg = np.vstack((E, Q))
    for i in range(Ng):
        bladeg[i] = fileGen+'/blade_'+str(i)
    TWQg = og.objectiveFunction(Ng, Pg, bladeg, scaler, mlp, vae, flow, paramBEMT)

    fn, v, d = og.distanceX(TWQg[:,1], TWQg[:,0], opt['Tmin'], TWQg[:,2], TWQg[:,3], opt['Qtol'])
    p = og.twoPenalties(fn, v)
    fp = p+d
    Metrics = og.updateMetrics(Metrics, TWQg, fp)
    NEF += Ng
    print(f"Gen {g} | W_current = {Metrics['W_opt'][-1]:.2f}")
    if NEF >= 4000:
        break
    if og.check_convergence(Metrics['W_opt'], threshold=0.001, patience=10):
        break
        
    
#%% Results
print('Optimization completed \n')
pp.printPlots(g, Metrics, opt['testDir'])
pp.saveBladeOpt(fp, bladeg, 'tests/'+opt['testDir'])
#%%
pp.blade3D('tests/'+opt['testDir'])

timeCPU = time.time() - t0

