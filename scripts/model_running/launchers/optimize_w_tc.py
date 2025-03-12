import numpy as np
import sys

from scipy.optimize import minimize 

sys.path.append("../../../con-con-models/")

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.modelanalysis.model as md 
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl 
import ccmodels.utils.watermark as wtm

simid = int(sys.argv[1])
bL23  = float(sys.argv[2])
bL4   = float(sys.argv[3])
savefolder = sys.argv[2]

datafolder = "data"

nreps = 1

units, connections, rates = loader.load_data(prepath=datafolder, orientation_only=True)
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

orionly= True
local_connectivity = False 
mode = 'cosine'

N = 8000
kee = 400

l23 = fl.filter_neurons(units, layer='L23', tuning='matched') 
l23id = l23['id'].values

datacurve = np.mean(dutl.shift_multi(rates[l23id, :], l23['pref_ori']), axis=0) 

def dosim(pars, args):
    tuning_curve = np.zeros(8)
    J,g,theta,sigma,hEI,hII=pars 
    thetaE, thetaI = theta, theta
    sigmaE, sigmaI = sigma, sigma

    datacurve = args[0]

    for j in range(nreps):
        aE_t, re, ri, rx, std_re, units_sample, QJ = md.make_simulation_cluster(units, connections, rates, kee, N, J, g, thetaE, thetaI, sigmaE, sigmaI, hEI=hEI, hII=hII, mode=mode, prepath=datafolder, orionly=orionly, cos_b=[bL23, bL4, 0., 0., 0., 0.])
        

        neurons_L23 = fl.filter_neurons(units_sample, layer='L23', cell_type='exc')
        tuning_curve += np.mean(dutl.shift_multi(re, neurons_L23['pref_ori']), axis=0) 

    tuning_curve /=  nreps

    error = np.sum((tuning_curve-datacurve)**2)
    print(pars, tuning_curve, error)

    return error 


J = 1.0 + 3*np.random.rand()
g = 3*np.random.rand()
theta = 15 + 5*np.random.rand()
sigma = 7 + 5*np.random.rand()
hEI = 50 + 100*np.random.rand()
hII = 100 + 400*np.random.rand()

pars0     = [J, g, theta, sigma, hEI, hII]
result    = minimize(dosim, pars0, args=[datacurve], method='Nelder-Mead', tol=1e-3)
simerror  = dosim(result.x, [datacurve])
header = wtm.add_metadata(extra="Optimized with TC only, parameters use theta and sigma.")
np.savetxt(f"{datafolder}/model/{savefolder}/metadata{simid}", [], header=header)
np.save(f'{datafolder}/model/{savefolder}/shortoptiresultsingle{simid}_pars',  result.x)
np.save(f'{datafolder}/model/{savefolder}/shortoptiresultsingle{simid}_error', simerror)
