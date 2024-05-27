import numpy as np
import torch
import pickle
import sbi.utils as sbiut
import sbi.inference as sbinfer
from sbi import analysis as sbians
from os import listdir
from os.path import isfile, join

from KDEpy import FFTKDE

def get_estimation_parameters(post_samples, prior_space):
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
    most_common = []

    #For each parameter, do the thing
    for i in range(post_samples.shape[1]):
        #Make the histogram from the samples, using the prior as binning
        hist, edges = np.histogram(post_samples[:,i], bins=prior_space[i], density=True)
        centered = 0.5*(edges[1:]+edges[:-1])

        #Get most common parameters
        most_common.append(centered[ np.argmax(hist)])
    
    return torch.tensor(most_common)



def get_estimation_parameters_KDE(post_samples, bw='ISJ'):
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
    most_common = []

    #For each parameter, do the thing
    for i in range(post_samples.shape[1]):
        x, y = FFTKDE(kernel='gaussian', bw=bw).fit(post_samples[:,i].numpy()).evaluate()
        most_common.append(x[np.argmax(y)])
    
    return torch.tensor(most_common)


def prepare_priors(j_int, g_int, th_int, s_int, nintervals=1000):
    j0, jf = j_int 
    g0, gf = g_int 
    theta0, thetaf = th_int 
    sigma0, sigmaf = s_int 

    prior_lowbound = torch.tensor([j0, g0, theta0, sigma0])
    prior_highbound = torch.tensor([jf, gf, thetaf, sigmaf])

    intervals = [j_int, g_int, th_int, s_int]

    prior = sbiut.BoxUniform(low=prior_lowbound, high=prior_highbound)
    prior_space = np.array([np.linspace(inter[0], inter[1], nintervals) for inter in intervals])

    return prior, prior_space, intervals



def train_network(prior, params, summary_stats):
    inference = sbinfer.SNPE(prior=prior)
    inference = inference.append_simulations(params.float(), summary_stats.float())
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    return posterior

def save_network(posterior, savefile):
    with open(savefile, "wb") as handle:
        pickle.dump(posterior, handle)

def load_network(path):
    with open(path, "rb") as handle:
        posterior = pickle.load(handle)
    return posterior