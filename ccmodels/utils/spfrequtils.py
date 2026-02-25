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

    d = post - pre
    max_angle = nangles//4 if half else nangles//2

    if d <= -max_angle:
        return d + 2*max_angle
    elif d > max_angle:
        return d - 2*max_angle
    else:
        return d

def signed_dist_vectorized(pre, post, nangles=16, half=True):
    """
    Computes a signed difference between pre a post, by taking into account periodic boundaries.
    In this way, we get differences in [-k, ..., 0, ...k], being nangle-k mapped to -k until -nangle//2,
    where results jump to be positive. 
    """
    dtheta = post - pre
    max_angle = nangles//4 if half else nangles//2

    mask1 = dtheta <= -max_angle
    mask2 = dtheta > max_angle

    dtheta[mask1] = dtheta[mask1] + 2*max_angle
    dtheta[mask2] = dtheta[mask2] - 2*max_angle

    return dtheta





def unsigned_dist(pre, post, nangles=16, half=True):
    """
    Classic distance with boundary conditions between angles pre and post, given as integers.
    """
    d = abs(post - pre)
    max_angle = nangles//2 if half else nangles
    return np.minimum(d, max_angle - d)

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
