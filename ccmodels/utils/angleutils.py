import numpy as np
import pandas as pd

def angle_indexer(pref_orientation):
    '''This function returns the index of the preferred orientation in a 16 bin discretization of the orientation space
    Args:
    pref_orientation: float, preferred orientation of the neuron
    
    Returns:
    indexed_angle: int, index of the preferred orientation in the 16 bin discretization of the orientation space
    '''
    #indexed_angle = int(round(pref_orientation/round(((2*np.pi)/16),8), 0))
    return (pref_orientation * 8 / np.pi).astype(int) 


def constrain_angles(thetas, nangles=16, negatives=True):
    """
    Constrain the angle indices to be in [0, nangles], which is
    sometimes necessary to operate
    """
    new_thetas = thetas.copy()
    
    #Negatives becomes 16 - X
    if negatives:
        negative = thetas< 0 
        new_thetas[negative] = nangles + thetas[negative] #We put a + because they are already negative

    #Large ones bounded in [0, 16]
    large = np.abs(thetas) >= nangles 
    new_thetas[large] = np.sign(thetas[large]) * (thetas[large] % nangles) #Python modulo always return positive, so add the sign manually 

    return new_thetas


def signed_angle_dist(pre, post, nangles=16, half=True):
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

#TODO: this might definitely substitute the code in construct_delta_ori and maybe even depcreate the function above
def signed_angle_dist_vectorized(pre, post, nangles=16, half=True):
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





def angle_dist(pre, post, nangles=16, half=True):
    """
    Classic distance with boundary conditions between angles pre and post, given as integers.
    """
    d = abs(post - pre)
    max_angle = nangles//2 if half else nangles
    return np.minimum(d, max_angle - d)


def construct_delta_ori(v1_neurons, v1_connections, nangles=16, half=True):
    """
    Given the tables of neurons and connections, get the array of delta orientations for each link and returns it.
    """

    #Get the indices of the pre and post neurons for each connection
    id_pre = v1_connections["pre_id"]
    id_post = v1_connections["post_id"]

    #Then we grab the angles for each connection
    angles_pre = v1_neurons.loc[id_pre, "pref_ori"].values 
    angles_post = v1_neurons.loc[id_post, "pref_ori"].values

    #Compute the difference 
    #The code that follows below is a generalization of angle_diff that's fast for vectors
    dtheta = angles_post - angles_pre 

    #What is the maximumf difference? Depeds if we are using orientation-only
    max_angle = nangles//4 if half else nangles//2

    #Boundary conditions
    mask1 = dtheta <= -max_angle
    mask2 = dtheta > max_angle

    dtheta[mask1] = dtheta[mask1] + 2*max_angle
    dtheta[mask2] = dtheta[mask2] - 2*max_angle

    return dtheta