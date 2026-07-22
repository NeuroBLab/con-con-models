import numpy as np
import sys
import os

os.environ['USE_FREQ'] = 'true'

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

from scipy.stats import skew

simid = int(sys.argv[1])
sample_mode = sys.argv[2] #normal, tunedinh, kin 
savefolder = sys.argv[3]
sbinet = sys.argv[4]
fixed_kee = int(sys.argv[5]) 

datafolder = "data"

def compute_prob_dif_layer(units, connections, layer):
    p = np.zeros(15)
    p_err = np.zeros(15)

    ndif = np.zeros(15)

    for kpre in range(8):
        pre_neurons  = fl.filter_neurons(units, layer=layer, tuning='tuned', proofread='ax_clean')
        pre_neurons  = pre_neurons.loc[pre_neurons['pref_ori']==kpre, ['id']].rename(columns = lambda x : f"pre_{x}")
        for kpost in range(8):

            dif = kpre - kpost

            post_neurons = fl.filter_neurons(units, layer='L23', tuning='tuned')
            post_neurons = post_neurons.loc[post_neurons['pref_ori']==kpost, ['id']].rename(columns = lambda x : f"post_{x}")

            pre_neurons['key'] = 1
            post_neurons['key'] = 1
            pairs = pre_neurons.merge(post_neurons, on='key')[['pre_id', 'post_id']]
            selected_connections = fl.synapses_by_id(connections, pre_ids=pre_neurons['pre_id'], post_ids=post_neurons['post_id'], who='both')

            p[dif + 7] += len(selected_connections) / len(pairs)
            p_err[dif + 7] += np.sqrt(p[dif + 7] * (1 - p[dif + 7]) / len(pairs)) 
            ndif[dif + 7] += 1

    p /= ndif
    p_err /= ndif

    maxp = np.max(p) 
    p /= maxp
    p_err /= maxp

    return p, p_err

def compute_conn_prob(units, connections):
    p = {}
    p_err = {}
    
    for layer in ['L23', 'L4']:
        p[layer], p_err[layer] = compute_prob_dif_layer(units, connections, layer)
    
    return p, p_err

units, connections, rates = loader.load_data(prepath=datafolder, orientation_only=True)
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

orionly= True
local_connectivity = False 
mode = 'cosine'

N = 20 * fixed_kee + 1
N_2save = 200

def dosim(pars):
    tuning_curve = np.zeros(15)
    conprob      = np.zeros(15)
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
        #utl.write_synthetic_data(f"testrandom{simid}", units_sample, connections_sample, re, ri, rx, original_prefori, prepath=datafolder)
        #units_sample, connections_sample, rates_sample, n_neurons, target_prefori = utl.load_synthetic_data(f"testrandom{simid}", prepath=datafolder)
        rates_sample = utl.format_synthetic_data_4conprob(units_sample, connections_sample, re, ri, rx)
        conprob = compute_conn_prob(units_sample, connections_sample)
    else:
        trivial_conprob= np.array([0.]*8  + [1.] + [0.]*6)
        conprob = {'L23':trivial_conprob, 'L4':trivial_conprob}

    return tuning_curve, conprob, re


nsims = 1000 
n_experiments = 1

if len(sbinet) < 5:

    best_pars = np.array([[9.98869781e+01, 4.66699005e+02, 2.98812389e-01, 1.46130979e-01]])
    
    unos = np.ones(nsims)

    J      = 7.92474210e-01 * unos  
    g      = 3.59854251e-01 * unos
    sigmaE = 7.69989061e+00 * unos
    sigmaI = 7.60365248e+00 * unos
    hEI = 50 + 200*np.random.rand(nsims)
    hII = 100 + 500*np.random.rand(nsims)
    b23 = 0.1 + 0.6*np.random.rand(nsims) 
    b4  = 0.1 + 0.6*np.random.rand(nsims)     

    if sample_mode == 'kin':
        kee = 30 + 570*np.random.rand(nsims)
    else:
        kee = fixed_kee * unos 

    header = wtm.add_metadata(extra="Using random betas, single run for each network. Using spatial frequency. Sample mode = {sample_mode}")
else:
    nsims = 100 
    n_experiments = 10
    posterior = msbi.load_posterior(f"{datafolder}/model/sbi_networks/{sbinet}") 

    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    neurons_L4 = fl.filter_neurons(units, layer='L4', tuning='matched')

    rates23 = rates[neurons_L23['id'], :]
    tcurvedata = np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0)
    means_data = compute_conn_prob(units, connections)

    #Take the tuning curve at three points: start, minimum, and end 
    cvo, cvd = utl.compute_circular_variance(tcurvedata, orionly=True)
    r0 = tcurvedata[0] 
    rf = tcurvedata[-1] 

    #Connection probability reduction at beginning and end for L23 adn L4
    #pL23 = 0.5 * (means_data['L23'][0] + means_data['L23'][-1]) 
    #pL4  = 0.5 * (means_data['L4'][0] + means_data['L4'][-1]) 

    pL23 = means_data['L23'][0] 
    pL4  = means_data['L4'][0] 

    pL23mid = means_data['L23'][2] 
    pL4mid  = means_data['L4'][2] 

    summary_data = np.zeros(12)
    summary_data[0] = r0
    summary_data[1] = rf
    summary_data[2] = cvd 
    summary_data[3] = pL23 
    summary_data[4] = pL4 
    summary_data[5] = pL23mid 
    summary_data[6] = pL4mid 

    cvoexp, cvdexp = utl.compute_circular_variance(rates23, orionly=True)

    #Before it was from 5
    summary_data[7] = np.mean(cvdexp)
    summary_data[8] = np.std(cvdexp)
    summary_data[9] = skew(cvdexp) 

    logrates = np.log(rates23.flatten()) 
    summary_data[10] = np.mean(logrates) 
    summary_data[11] = np.std(logrates) 
    
    summary_data = torch.tensor(summary_data)

    posterior_samples = posterior.sample((nsims,), x=summary_data.float()).numpy()

    if sample_mode == 'kin':
        J,g,sigmaE,sigmaI,hEI,hII,kee = np.transpose(posterior_samples) 
        b23 = np.zeros(J.shape)
        b4  = np.zeros(J.shape)
    else:
        J,g,sigmaE,sigmaI,hEI,hII,b23,b4 = np.transpose(posterior_samples) 
        kee = fixed_kee * np.ones(nsims)

    header = wtm.add_metadata(extra=f"SBI simulation using network {sbinet} with sample mode {sample_mode} and spatial frequency")


np.savetxt(f"{datafolder}/model/simulations/{savefolder}/metadata{simid}", [], header=header)
output = open(f'{datafolder}/model/simulations/{savefolder}/{simid}.txt', 'a')
warnings.simplefilter("ignore")
for i in range(nsims):
    pars     = [J[i], g[i], sigmaE[i], sigmaI[i], hEI[i], hII[i], b23[i], b4[i], kee[i]]
    for ix_exp in range(n_experiments):
        j = n_experiments * i + ix_exp 

        tcurve, conprob, re = dosim(pars)
        result = np.concatenate((pars, tcurve, conprob['L23'], conprob['L4']))
        np.savetxt(output, result[np.newaxis, :])

        idx_sample = np.random.choice(re.shape[0], N_2save)
        np.save(f'{datafolder}/model/simulations/{savefolder}/{simid}_rates{j}.npy', re[idx_sample]) 

output.close()
