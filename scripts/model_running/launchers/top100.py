import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings

import sys
import os 
sys.path.append(os.getcwd())
import argparse

import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.modelanalysis.currents as mcur
import ccmodels.modelanalysis.sbi_utils as msbi

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl

import ccmodels.utils.watermark as wtm

from sklearn.preprocessing import robust_scale

# ---------------- Aux functions for computations -----------------

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=1000):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=n_samps)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]
        #meandata[layer]  = plotutils.add_symmetric_angle(p['mean'].values)
        meandata[layer]  = p['mean'].values

    return meandata


def normalize_errors(*arrays):
    """Apply robust scaling (median-centered, MAD-scaled) to each error array."""
    return [robust_scale(arr.reshape(-1, 1)).flatten() for arr in arrays]

def compute_total_loss(error_arrays, weights=None):
    """Compute a weighted total loss from a list of normalized error arrays."""
    error_matrix = np.vstack(error_arrays).T
    if weights is None:
        weights = np.ones(error_matrix.shape[1])
    return error_matrix @ weights

def load_rates(simid, ratefolder, is_sbi = True):
    if is_sbi:
        rate_id = simid // 100
        part_id = simid - rate_id * 100
        return np.load(f"{ratefolder}/{rate_id}_rates{part_id}.npy")
    else:
        rate_id = simid // 1000
        part_id = simid - rate_id * 1000
        return np.load(f"{ratefolder}/{rate_id}_rates{part_id}.npy")


def get_best_params(datafolder, summary_data, ntops, is_sbi = True): 
    # Process rate data and compute distances
    sampled_rate_histograms = []
    sampled_cv_histograms = []

    rate_dists_L1 = []
    cv_dists_L1 = []

    nparams = 8
    params = np.empty((0, nparams))
    summary_stats = np.empty((0,24)) 

    nsims = 100
    for i in range(nsims):
        inputfile = np.loadtxt(f"{datafolder}/{i}.txt")
        if inputfile.size > 0:
            params = np.vstack((params, inputfile[:, :nparams]))
            summary_stats = np.vstack((summary_stats, inputfile[:, 9:]))
    ndata = params.shape[0]

    rate_bins = np.linspace(0, 10, 20)  
    cv_bins = np.linspace(0, 1, 20)    

    #avg_tuning_curves = [] 

    if is_sbi:
        n_experiments = 10
    else:
        n_experiments = 1

    # Loop over each simulation to extract the rate data and compute histograms
    for sim_idx in range(ndata//n_experiments):

        #Compute the first one!!!
        sampled_rates_here = load_rates(sim_idx * n_experiments, datafolder, is_sbi)
        
        _, cvd = utl.compute_circular_variance(sampled_rates_here, orionly=True)
        rate_hist, _ = np.histogram(sampled_rates_here.ravel(), bins=rate_bins, density=True)
        cv_hist, _ = np.histogram(cvd, bins=cv_bins, density=True)

        #Average the rest!!!
        for exp_ix in range(1, n_experiments):    

            sampled_rates_here = load_rates(sim_idx * n_experiments + exp_ix, datafolder, is_sbi)
            
            _, cvd = utl.compute_circular_variance(sampled_rates_here, orionly=True)
            rh, _ = np.histogram(sampled_rates_here.ravel(), bins=rate_bins, density=True)
            ch, _ = np.histogram(cvd, bins=cv_bins, density=True)

            rate_hist += rh
            cv_hist   += ch

        rate_hist /= n_experiments
        cv_hist   /= n_experiments

        # Compute L1 mean distances to data histograms 
        rate_dist = np.abs(rate_hist - summary_data['ratehist']).mean()
        cv_dist   = np.abs(cv_hist - summary_data['cvhist']).mean()

        #sampled_rate_histograms.append(rate_hist)
        #sampled_cv_histograms.append(cv_hist)

        rate_dists_L1.append(rate_dist)
        cv_dists_L1.append(cv_dist)
        
        # Compute average tuning curve after shifting each neuron's curve
        #avg_tuning_curve = np.mean(dutl.shift_multi(sampled_rates_here, np.argmax(sampled_rates_here, axis=1)), axis=0)
        #avg_tuning_curves.append(avg_tuning_curve)


    # Convert the lists to arrays for further analysis
    #sampled_rate_histograms = np.array(sampled_rate_histograms)
    #sampled_cv_histograms = np.array(sampled_cv_histograms)

    summary_stats = summary_stats.reshape((summary_stats.shape[0]//n_experiments, n_experiments, summary_stats.shape[1]))
    tcurve_sim=summary_stats[:, :, :8].mean(axis=1)
    P_conn23_sim=summary_stats[:, :, 8:16].mean(axis=1)
    P_conn4_sim=summary_stats[:, :, 16:].mean(axis=1)


    rate_dists_L1 = np.array(rate_dists_L1)
    cv_dists_L1 = np.array(cv_dists_L1)
    avg_tuning_curves_L1 = np.mean(np.abs(tcurve_sim- summary_data['tcurve']),axis=1)
    avg_P_conn23_L1 = np.mean(np.abs(P_conn23_sim- summary_data['pL23']),axis=1)
    avg_P_conn4_L1 = np.mean(np.abs(P_conn4_sim- summary_data['pL4']),axis=1)

    # Normalize
    norm_errors = normalize_errors(avg_tuning_curves_L1, 
                                rate_dists_L1,
                                cv_dists_L1, 
                                avg_P_conn23_L1, 
                                avg_P_conn4_L1)

    # Compute total loss
    weights = np.array([10, 1., 2, 1, 1])

    total_loss = compute_total_loss(norm_errors, weights=weights)

    # Sort
    sorted_idx = np.argsort(total_loss)
    top_sims = sorted_idx[:ntops]

    return params[top_sims] 

def dosim(pars,kee):
    J,g,sigmaE,sigmaI,hEI,hII,bL23,bL4=pars 

    if sample_mode == 'kin':
        cos_modulation = np.zeros(6) 
    elif sample_mode == 'tunedinh':
        cos_modulation = [bL23, bL4, bL23, bL23, bL23, bL4]
    else:
        cos_modulation = [bL23, bL4, 0., 0., 0., 0.] 

    cv = np.empty(4)
    aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, kee, N, J, g, hEI=hEI, hII=hII,theta_E=20., sigma_tE=sigmaE, theta_I=20.0, sigma_tI=sigmaI, cos_b=cos_modulation, mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath=datafolder)

    _, cvsim = utl.compute_circular_variance(re, orionly=True)    
    cv[0] = cvsim.mean()

    cvidx = 1
    for reshuffle_mode in ['all', 'L23', 'L4']:
        aE_t, aI_t, re, ri, stdre, units_reshuffle, connections_reshuffle, QJ_reshuffle = md.make_simulation_fixed_structure(units_sample, connections_sample, QJ, rx, n_neurons, theta_E=20., sigma_tE=sigmaE, hEI=hEI, hII=hII, theta_I=20., sigma_tI=sigmaI, orionly=True, reshuffle=reshuffle_mode)

        _, cvsim = utl.compute_circular_variance(re, orionly=True)    
        cv[cvidx] = cvsim.mean()
        cvidx += 1

    return cv 


sample_mode = sys.argv[1] #normal, tunedinh, kin 
savefolder = sys.argv[2]
fixed_kee = int(sys.argv[3]) 
is_sbi = True
datafolder = "data"

units, connections, rates = loader.load_data(prepath=datafolder, orientation_only=True)
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

orionly= True
local_connectivity = False 
mode = 'cosine'

N = 20 * fixed_kee 
N_2save = 200

ntop = 100 

#Get all the data and store in a dict
summary_data = {}
rate_bins = np.linspace(0, 10, 20)  
cv_bins = np.linspace(0, 1, 20)    
rate_bin_centers = 0.5 * (rate_bins[1:] + rate_bins[:-1])
cv_bin_centers = 0.5 * (cv_bins[1:] + cv_bins[:-1])


neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
rates23 = rates[neurons_L23['id'], :]
summary_data['tcurve'] = np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0)
summary_data['ratehist'], _ = np.histogram(rates23.ravel(), bins=rate_bins, density=True)
_, cvdexp = utl.compute_circular_variance(rates, orionly=True)
summary_data['cvhist'], _ = np.histogram(cvdexp, bins=cv_bins, density=True)


means_data = compute_conn_prob(units, connections)
summary_data['pL23'] = means_data['L23']
summary_data['pL4']  = means_data['L4']


if sample_mode == 'normal':
    #params = get_best_params("data/model/simulations/sbi_randowkMINI", summary_data, ntop) 
    params = get_best_params("data/model/simulations/sbi_randk150", summary_data, ntop, is_sbi=is_sbi) 
else:
    #params = get_best_params("data/model/simulations/sbi_tunedlowkMINI", summary_data, ntop) 
    params = get_best_params("data/model/simulations/sbi_tunedk150", summary_data, ntop, is_sbi = is_sbi) 

J,g,sigmaE,sigmaI,hEI,hII,b23,b4 = np.transpose(params) 

header = wtm.add_metadata(extra="Circ Var of the 100 SBI best simulations. Sample mode = {sample_mode}")
np.savetxt(f"{datafolder}/model/simulations/{savefolder}/metadata", [], header=header)

cvresults = np.empty((ntop, 4))
warnings.simplefilter("ignore")
for i in range(ntop):
    pars     = params[i,:] #[J[i], g[i], sigmaE[i], sigmaI[i], hEI[i], hII[i], b23[i], b4[i], kee[i]]
    cvresults[i, :] = dosim(pars, fixed_kee)
np.savetxt(f'{datafolder}/model/simulations/{savefolder}/top100.txt', cvresults) 

