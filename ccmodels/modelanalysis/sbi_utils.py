import torch
import sbi.utils as sbiut
import sbi.inference as sbinfer
from sbi import analysis as sbians
from os import listdir
from os.path import isfile, join

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.modelanalysis.utils as utl
import ccmodels.modelanalysis.currents as mcur
import ccmodels.dataanalysis.filters as fl

import numpy as np
import pandas as pd
import pickle

from KDEpy import FFTKDE

def get_simulations_summarystats(path, features, average_disorder=False, nsims=None):
    path = "k400_disorder"
    average_disorder = False 
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    params = np.empty((0,4)) 
    summary_stats = np.empty((0, len(features))) 
    for file in onlyfiles:
        filecontent = pd.read_csv(f'{path}/{file}')

        p = filecontent[['J', 'g', 'theta', 'sigma']]
        sumstats = filecontent[features]

        if average_disorder:
            p        = p.groupby(np.arange(len(p))//10).mean()
            sumstats = sumstats.groupby(np.arange(len(sumstats))//10).mean()
        
        params = np.vstack([params, p])
        summary_stats = np.vstack([summary_stats, sumstats]) 

    if nsims==None:
        return torch.tensor(params), torch.tensor(summary_stats)
    else:
        sims = np.arange(100000)
        ndata = nsims*10
        np.random.shuffle(sims)
        sims = sims[:ndata]
        params = torch.tensor(params[sims])
        summary_stats = torch.tensor(summary_stats[sims]).float()

        return params, summary_stats

def get_data_summarystats(features, prepath=".", orionly=True):
    orionly = True
    units, connections, rates = loader.load_data(orientation_only=True, prepath=prepath) 

    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    osi_e = utl.compute_orientation_selectivity_index(rates[units_e['id']])
    rates_e = rates[units_e['id']]

    cveo, cved = utl.compute_circular_variance(rates[units_e['id']], orionly=True)

    matchedunits = fl.filter_neurons(units, tuning='matched')
    matchedconnections = fl.synapses_by_id(connections, pre_ids=matchedunits['id'], post_ids=matchedunits['id'], who='both')
    vij = loader.get_adjacency_matrix(matchedunits, matchedconnections)
    currents = mcur.bootstrap_mean_current(units, vij, rates, tuning=['matched', 'matched'], cell_type=['exc', 'exc'])
    totalmean = currents['Total'].mean(axis=0).max()
    cvl23o, cvl23d = utl.compute_circular_variance(currents['L23'].mean(axis=0)/totalmean, orionly=orionly)
    cvl4o, cvl4d = utl.compute_circular_variance(currents['L4'].mean(axis=0)/totalmean, orionly=orionly)

    summary_data = {'mean_re' : [rates.mean()], 'std_re': [rates_e.std()], 'mean_cve_dir': [cved.mean()], 'std_cve_dir':[cved.std()],
                    'cv_curl23': [cvl23d], 'cv_curl4':[cvl4d], 'indiv_traj_std':[0.]}
    summary_data = pd.DataFrame(summary_data)
    
    return torch.tensor(summary_data[features].values)

def setup_prior():
    j0, jf = 0, 5
    #g0, gf = 1., 7.
    g0, gf = 1., 5.
    theta0, thetaf = 10, 25
    #sigma0, sigmaf = 1, 20 
    sigma0, sigmaf = 1, 15 

    prior_lowbound = torch.tensor([j0, g0, theta0, sigma0])
    prior_highbound = torch.tensor([jf, gf, thetaf, sigmaf])

    intervals = [[j0, jf], [g0, gf], [theta0, thetaf], [sigma0, sigmaf]]

    prior = sbiut.BoxUniform(low=prior_lowbound, high=prior_highbound)
    prior_space = np.array([np.linspace(inter[0], inter[1], 1000) for inter in intervals])
    return prior, intervals


def train_sbi(prior, params, summary_stats):
    inference = sbinfer.SNPE(prior=prior)
    inference = inference.append_simulations(params.float(), summary_stats.float())
    density_estimator = inference.train()
    return inference.build_posterior(density_estimator)


def save_posterior(path, posterior):
    with open(path, "wb") as handle:
        pickle.dump(posterior, handle)

def load_posterior(path):
    with open(path, "rb") as handle:
        posterior = pickle.load(handle)
    return posterior

#This function is used to estimate the most probable parameters given posterior samples
def get_estimation_parameters(post_samples, npars, bw='ISJ', joint_posterior=True):
    """
    This function gets the posterior samples and returns the most common (distribution max)
    Parameters
    ==========
    - post_samples : ndarray
        Samples of the parameters obtained from SBI's posterior.sample
    - npars : int
        Number of parameters of the model
    - prior_space : npar*M ndarray
        Contains the binning for each one of the parameters. Bounds should coincide with prior ones for best results
    """

    if joint_posterior:
        kde = FFTKDE()
        grid_points = 64 
        post = kde.fit(post_samples.numpy())
        grid, postev =  post.evaluate(grid_points)
        return grid[np.argmax(postev), :]
    else:
        most_common = []

        #For each parameter, do the thing
        for i in range(npars):
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(post_samples[:,i].numpy()).evaluate()
            most_common.append(x[np.argmax(y)])
        
        return torch.tensor(most_common)