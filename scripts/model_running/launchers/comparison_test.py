import numpy as np
import sys
import os

sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl 
import ccmodels.dataanalysis.statistics_extraction as ste

numpyseed = int(sys.argv[1])
np.random.seed(numpyseed)

datafolder = "data"

#Heat up RNG
np.random.rand(100)

bL23 = 0.4
bL4  = 0.2

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=1000):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=n_samps)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]
        meandata[layer]  = p['mean'].values

    return meandata 

units, connections, rates = loader.load_data(prepath=datafolder, orientation_only=True)
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

orionly= True
local_connectivity = False 
mode = 'cosine'

N = 3000
kee = 150 

l23 = fl.filter_neurons(units, layer='L23', tuning='matched') 
l23id = l23['id'].values

datacurve = np.mean(dutl.shift_multi(rates[l23id, :], l23['pref_ori']), axis=0) 
dataprobs = compute_conn_prob(units, connections)

def dosim(pars):
    tuning_curve = np.zeros(8)
    conprob      = np.zeros(8)
    J,g,sigmaE,sigmaI,hEI,hII=pars 

    aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, kee, N, J, g, hEI=hEI, hII=hII,theta_E=20., sigma_tE=sigmaE, theta_I=20.0, sigma_tI=sigmaI, cos_b=[bL23, bL4, bL23, bL23, bL23, bL4], mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath=datafolder)

    neurons_L23 = fl.filter_neurons(units_sample, layer='L23', cell_type='exc')
    tuning_curve += np.mean(dutl.shift_multi(re, neurons_L23['pref_ori']), axis=0) 

    if len(units_sample.pref_ori.unique()) == 8:
        utl.write_synthetic_data(f"unittest", units_sample, connections_sample, re, ri, rx, original_prefori, prepath=datafolder)
        units_sample, connections_sample, rates_sample, n_neurons, target_prefori = utl.load_synthetic_data(f"unittest", prepath=datafolder)
        conprob += compute_conn_prob(units_sample, connections_sample, n_samps=1)['L23']
    else:
        conprob += np.array([0.,0.,0.,1.,0.,0.,0.,0.])

    return tuning_curve, conprob 


J = 1.0 + 3*np.random.rand()
g = 5*np.random.rand()
sigmaE = 7 + 5*np.random.rand()
sigmaI = 7 + 5*np.random.rand()
hEI = 50 + 100*np.random.rand()
hII = 100 + 400*np.random.rand()

pars0     = [J, g, sigmaE, sigmaI, hEI, hII]
tc, cp    = dosim(pars0)

print("Random seed")
print(numpyseed)
print("Data tuning curve")
print(datacurve)
print("Data conn. probability")
print(dataprobs)
print("Selected parameters")
print(pars0)
print("Simulation tuning curve")
print(tc)
print("Simulation conn. probability")
print(cp)

