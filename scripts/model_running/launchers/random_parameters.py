import numpy as np
import sys
import os

import warnings

sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl 
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.modelanalysis.sbi_utils as msbi
import ccmodels.utils.watermark as wtm
import torch

simid = int(sys.argv[1])
sample_mode = sys.argv[2] #normal, tunedinh, kin 
savefolder = sys.argv[3]
sbinet = sys.argv[4]
fixed_kee = int(sys.argv[5]) 

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


units, connections, rates = loader.load_data(prepath=datafolder, orientation_only=True)
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

orionly= True
local_connectivity = False 
mode = 'cosine'

N_2save = 200
N = 20 * fixed_kee 


def dosim(pars):
    tuning_curve = np.zeros(8)
    conprob      = np.zeros(8)
    J,g,sigmaE,sigmaI,hEI,hII,bL23,bL4,kee=pars 

    if sample_mode == 'kin':
        cos_modulation = np.zeros(6) 
    elif sample_mode == 'tunedinh':
        cos_modulation = [bL23, bL4, bL23, bL23, bL23, bL4]
    else:
        cos_modulation = [bL23, bL4, 0., 0., 0., 0.] 

    aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, kee, N, J, g, hEI=hEI, hII=hII,theta_E=20., sigma_tE=sigmaE, theta_I=20.0, sigma_tI=sigmaI, cos_b=cos_modulation, mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath=datafolder)

    neurons_L23 = fl.filter_neurons(units_sample, layer='L23', cell_type='exc')
    tuning_curve += np.mean(dutl.shift_multi(re, neurons_L23['pref_ori']), axis=0) 

    if neurons_L23.pref_ori.nunique() == 8:
        utl.write_synthetic_data(f"testrandom{simid}", units_sample, connections_sample, re, ri, rx, original_prefori, prepath=datafolder)
        #units_sample, connections_sample, rates_sample, n_neurons, target_prefori = utl.load_synthetic_data(f"testrandom{simid}", prepath=datafolder)
        rates_sample = utl.format_synthetic_data_4conprob(units_sample, connections_sample, re, ri, rx)
        conprob = compute_conn_prob(units_sample, connections_sample, n_samps=1)
    else:
        trivial_conprob= np.array([0.,0.,0.,1.,0.,0.,0.,0.])
        conprob = {'L23':trivial_conprob, 'L4':trivial_conprob}

    return tuning_curve, conprob, re


nsims = 1000 

if len(sbinet) < 5:
    J = 4*np.random.rand(nsims)
    g = 5*np.random.rand(nsims)
    sigmaE = 7 + 5*np.random.rand(nsims)
    sigmaI = 7 + 5*np.random.rand(nsims)
    hEI = 50 + 100*np.random.rand(nsims)
    hII = 100 + 400*np.random.rand(nsims)
    b23 = 0.1 + 0.5*np.random.rand(nsims) 
    b4  = 0.1 + 0.5*np.random.rand(nsims)     

    if sample_mode == 'kin':
        kee = 30 + 570*np.random.rand(nsims)
    else:
        kee = fixed_kee * np.ones(nsims)

    header = wtm.add_metadata(extra="Using random betas, single run for each network. Sample mode = {sample_mode}")
else:
    nsims = 100 
    posterior = msbi.load_posterior(f"{datafolder}/model/sbi_networks/{sbinet}") 

    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    neurons_L4 = fl.filter_neurons(units, layer='L4', tuning='matched')

    rates23 = rates[neurons_L23['id'], :]
    tcurvedata = np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0)
    means_data = compute_conn_prob(units, connections)

    #For CV + rate dist
    cvoexp, cvdexp = utl.compute_circular_variance(rates23, orionly=True)

    bins = np.linspace(0.,1.,9)
    cvdist, _ = np.histogram(cvdexp, bins=bins, density=True)

    bins = np.linspace(0.,2.,9)
    rdist = np.zeros(8)
    nsampled = 200
    for nrep in range(100):
        idx = np.random.choice(rates23.shape[0], nsampled)
        distsample, _ = np.histogram(rates23[idx, :].ravel(), bins=bins, density=False)
        rdist += distsample


    rdist /= 100

    

    #summary_data = np.concatenate((tcurvedata, means_data['L23'], means_data['L4']))
    summary_data = np.concatenate((means_data['L23'], means_data['L4'], cvdist, rdist))
    summary_data = torch.tensor(summary_data)


    posterior_samples = posterior.sample((nsims,), x=summary_data.float()).numpy()

    if sample_mode == 'kin':
        J,g,sigmaE,sigmaI,hEI,hII,kee = np.transpose(posterior_samples) 
        b23 = np.zeros(J.shape)
        b4  = np.zeros(J.shape)
    else:
        J,g,sigmaE,sigmaI,hEI,hII,b23,b4 = np.transpose(posterior_samples) 
        kee = fixed_kee * np.ones(nsims)

    header = wtm.add_metadata(extra=f"SBI simulation using network {sbinet} with sample mode {sample_mode}")


np.savetxt(f"{datafolder}/model/simulations/{savefolder}/metadata{simid}", [], header=header)
output = open(f'{datafolder}/model/simulations/{savefolder}/{simid}.txt', 'a')
warnings.simplefilter("ignore")
for i in range(nsims):
    pars     = [J[i], g[i], sigmaE[i], sigmaI[i], hEI[i], hII[i], b23[i], b4[i], kee[i]]
    tcurve, conprob, re = dosim(pars)
    result = np.concatenate((pars, tcurve, conprob['L23'], conprob['L4']))
    np.savetxt(output, result[np.newaxis, :])

    idx_sample = np.random.choice(re.shape[0], N_2save)
    np.save(f'{datafolder}/model/simulations/{savefolder}/{simid}_rates{i}.npy', re[idx_sample]) 

output.close()
