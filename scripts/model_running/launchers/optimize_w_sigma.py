import numpy as np
import sys

from scipy.optimize import minimize 

sys.path.append("../../../con-con-models/")

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl 
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.utils.watermark as wtm

simid = int(sys.argv[1])
bL23  = float(sys.argv[2])
bL4   = float(sys.argv[3])
savefolder = sys.argv[4]

datafolder = "data"

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
dataprobs = compute_conn_prob(units, connections)

meandcurve2 = np.mean(datacurve)**2
meandprobs2 = np.mean(dataprobs['L23'])**2


def dosim(pars, args):
    tuning_curve = np.zeros(8)
    conprob      = np.zeros(8)
    J,g,sigmaE,sigmaI,hEI,hII=pars 

    datacurve = args[0]
    dataprobs = args[1]
    meandcurve2 = args[2]
    meandprobs2 = args[3]

    for j in range(nreps):
        aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, kee, N, J, g, hEI=hEI, hII=hII,theta_E=19., sigma_tE=sigmaE, theta_I=19.0, sigma_tI=sigmaI, cos_b=[bL23, bL4, 0., 0., 0., 0.], mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath=datafolder)

        neurons_L23 = fl.filter_neurons(units_sample, layer='L23', cell_type='exc')
        tuning_curve += np.mean(dutl.shift_multi(re, neurons_L23['pref_ori']), axis=0) 

        if len(units_sample.pref_ori.unique()) == 8:
            utl.write_synthetic_data(f"test{simid}", units_sample, connections_sample, re, ri, rx, original_prefori, prepath=datafolder)
            units_sample, connections_sample, rates_sample, n_neurons, target_prefori = utl.load_synthetic_data(f"test{simid}", prepath=datafolder)
            conprob += compute_conn_prob(units_sample, connections_sample, n_samps=1)['L23']
        else:
            conprob += np.array([0.,0.,0.,1.,0.,0.,0.,0.])


    tuning_curve /=  nreps
    conprob /= nreps

    #error = np.sum((tuning_curve-datacurve)**2)
    #error = np.sum((tuning_curve-datacurve)**2)/meandcurve2 + np.sum((conprob - dataprobs)**2)/meandprobs2
    error = np.sum((tuning_curve-datacurve)**2) + 200.*np.sum((conprob - dataprobs)**2) 
    print(pars, tuning_curve, error)

    return error 


J = 1.0 + 3*np.random.rand()
g = 5*np.random.rand()
sigmaE = 7 + 5*np.random.rand()
sigmaI = 7 + 5*np.random.rand()
hEI = 50 + 100*np.random.rand()
hII = 100 + 400*np.random.rand()

pars0     = [J, g, sigmaE, sigmaI, hEI, hII]
result    = minimize(dosim, pars0, args=[datacurve, dataprobs['L23'], meandcurve2, meandprobs2], method='Nelder-Mead')
simerror  = dosim(result.x, [datacurve, dataprobs['L23'], meandcurve2, meandprobs2])

header = wtm.add_metadata(extra="Optimized using both TC and connection probability, using the sigma parameters for EI")
np.savetxt(f"{datafolder}/model/simulations/{savefolder}/metadata{simid}", [], header=header)
np.save(f'{datafolder}/model/simulations/{savefolder}/bL23_{bL23:.2f}_bL4_{bL4:.2f}_{simid}_pars',  result.x)
np.save(f'{datafolder}/model/simulations/{savefolder}/bL23_{bL23:.2f}_bL4_{bL4:.2f}_{simid}_error', simerror)
