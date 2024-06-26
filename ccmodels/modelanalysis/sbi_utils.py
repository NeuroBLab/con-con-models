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

def get_simulations_summarystats(path, features, average_disorder=False, nfiles_avgdis=10, nsims=None):
    #Get all files in the specified path
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    params = np.empty((0,4)) 
    summary_stats = np.empty((0, len(features))) 
    for file in onlyfiles:
        #Read the file
        filecontent = pd.read_csv(f'{path}/{file}')

        #Get the parameters and the desired summary statistics
        p = filecontent[['J', 'g', 'theta', 'sigma']]
        sumstats = filecontent[features]

        #Average over the disorder, when specified
        if average_disorder:
            p        = p.groupby(np.arange(len(p))//nfiles_avgdis).mean()
            sumstats = sumstats.groupby(np.arange(len(sumstats))//nfiles_avgdis).mean()
        
        #put together the new results
        params = np.vstack([params, p])
        summary_stats = np.vstack([summary_stats, sumstats]) 

    #If not specified, return all simulations
    if nsims==None:
        return torch.tensor(params), torch.tensor(summary_stats)
    else:
        #When specified, return a random number of parameters 
        total_n_sims = np.arange(summary_stats.shape[0])
        ndata = nsims*nfiles_avgdis

        #Shuffle
        np.random.shuffle(total_n_sims)
        total_n_sims = total_n_sims[:ndata]
        params = torch.tensor(params[total_n_sims])
        summary_stats = torch.tensor(summary_stats[total_n_sims]).float()

        return params, summary_stats

def get_data_summarystats(features, prepath="data/", orionly=True):
    #Read the data
    units, connections, rates = loader.load_data(orientation_only=orionly, prepath=prepath) 

    #Compute the rates
    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    rates_e = rates[units_e['id']]

    #Compute the circular variance
    cveo, cved = utl.compute_circular_variance(rates[units_e['id']], orionly=True)

    #Get the adjacency matrix
    matchedunits = fl.filter_neurons(units, tuning='matched')
    matchedconnections = fl.synapses_by_id(connections, pre_ids=matchedunits['id'], post_ids=matchedunits['id'], who='both')
    vij = loader.get_adjacency_matrix(matchedunits, matchedconnections)

    #Use adjacency matrix + rates to get all system's currents between matched E neurons
    currents = mcur.bootstrap_mean_current(units, vij, rates, tuning=['matched', 'matched'], cell_type=['exc', 'exc'])
    totalmean = currents['Total'].mean(axis=0).max()

    #Get the tuning of the current
    cvl23o, cvl23d = utl.compute_circular_variance(currents['L23'].mean(axis=0)/totalmean, orionly=orionly)
    cvl4o, cvl4d = utl.compute_circular_variance(currents['L4'].mean(axis=0)/totalmean, orionly=orionly)

    #Put in the same format as the simulations and return
    summary_data = {'mean_re' : [rates.mean()], 'std_re': [rates_e.std()], 'mean_cve_dir': [cved.mean()], 'std_cve_dir':[cved.std()],
                    'cv_curl23': [cvl23d], 'cv_curl4':[cvl4d], 'indiv_traj_std':[0.]}
    summary_data = pd.DataFrame(summary_data)
    
    return torch.tensor(summary_data[features].values)

def setup_prior():
    #Setup the prior values
    j0, jf = 0, 5
    g0, gf = 1., 5.
    theta0, thetaf = 10, 25
    sigma0, sigmaf = 1, 15 

    #For the sbi
    prior_lowbound = torch.tensor([j0, g0, theta0, sigma0])
    prior_highbound = torch.tensor([jf, gf, thetaf, sigmaf])
    prior = sbiut.BoxUniform(low=prior_lowbound, high=prior_highbound)

    #for plotting
    intervals = [[j0, jf], [g0, gf], [theta0, thetaf], [sigma0, sigmaf]]

    #Return the prior in the useful formats
    return prior, intervals


def train_sbi(prior, params, summary_stats):
    #Prepare the SBI in the selected prior
    inference = sbinfer.SNPE(prior=prior)

    #Put the simulations 
    inference = inference.append_simulations(params.float(), summary_stats.float())

    #Train and return
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

    #Compute the parameters with the full joint posterior
    if joint_posterior:
        kde = FFTKDE()
        grid_points = 64 
        post = kde.fit(post_samples.numpy())
        grid, postev =  post.evaluate(grid_points)
        return grid[np.argmax(postev), :]
    else:
        #Evaluate for each parameter
        most_common = []

        #For each parameter, do the thing
        for i in range(npars):
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(post_samples[:,i].numpy()).evaluate()
            most_common.append(x[np.argmax(y)])
        
        return most_common