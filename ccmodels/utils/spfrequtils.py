import numpy as np
import pandas as pd

def signed_dist(pre, post, nangles=16, half=True):
    """
    Computes a signed difference between pre a post, by taking into account periodic boundaries.
    In this way, we get differences in [-k, ..., 0, ...k], being nangle-k mapped to -k until -nangle//2,
    where results jump to be positive. 
    """

    #TODO might not be needed after all, since the construction of the dtheta does not call this
    #and outside we might be always filtering for fucntionally matched neurons
    if pd.isnull(pre) or pd.isnull(post):
        return np.nan 

    return post - pre

def signed_dist_vectorized(pre, post, nangles=16, half=True):
    """
    Computes a signed difference between pre a post, by taking into account periodic boundaries.
    In this way, we get differences in [-k, ..., 0, ...k], being nangle-k mapped to -k until -nangle//2,
    where results jump to be positive. 
    """
    return post - pre





def unsigned_dist(pre, post, nangles=16, half=True):
    """
    Classic distance with boundary conditions between angles pre and post, given as integers.
    """
    return abs(post - pre)

def construct_delta_ori(v1_neurons, v1_connections, nfreqs=8, half=True):
    """
    Given the tables of neurons and connections, get the array of delta spatial frequencies for each link and returns it.
    """
    #Get the indices of the pre and post neurons for each connection
    id_pre = v1_connections["pre_id"]
    id_post = v1_connections["post_id"]

    #Values for each connection. Assume a override on pref_ori
    angles_pre = v1_neurons.loc[id_pre, "pref_ori"].values 
    angles_post = v1_neurons.loc[id_post, "pref_ori"].values

    #Compute the difference 
    #The code that follows below is a generalization of angle_diff that's fast for vectors
    return angles_post - angles_pre 
