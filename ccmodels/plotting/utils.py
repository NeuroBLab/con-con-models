import pandas as pd
import numpy as np

# ------------------------------------------------------------
# ---------------------- PLOTTING HELP-----------------
# ------------------------------------------------------------

def get_angles(kind="normal", half=True, nangles=16):
    """
    Get an array with the correct angles, useful for the x-axis plotting.

    Parameters
    ==========
    kind : string
        Can be "normal" (default), "centered", or "diff". When normal, returns angles from [0, 2π[ (for
        all angles, half=False). Centered returns from [-π, π[, and diff yields [-2π, 2π[. 
    half : bool
        if True (default), the ranges explained above are divided by 2 
    nangles : int
        number of angles, 16 by default
    """

    if half:
        ndivisions = nangles//2
        maxangle = np.pi 
    else:
        ndivisions = nangles
        maxangle = 2*np.pi 


    if kind=="normal":
        return np.linspace(0, maxangle, ndivisions) 
    elif kind=="centered":
        return np.linspace(-maxangle/2, maxangle/2, ndivisions)
    elif kind=="diff":
        return np.linspace(-maxangle, maxangle, ndivisions + 1)

def get_xticks(ax, max=np.pi, half=True):
    """
    Get the xticks for the xaxis of angles.

    Parameters
    max : float
        Gives the limit of the xticks. It can be set to pi/2, pi or 2pi. The value should be given for the full, half=False case
    half : bool
        If True, will divide max by two to accomodate it to half angles. 
    """

    if half:
        max /= 2.0

    if max < np.pi:
        ax.set_xticks([-np.pi/2, 0, np.pi/2], ["-π/2", "0", "π/2"])
    elif max == np.pi:
        ax.set_xticks([-np.pi, 0, np.pi], ["-π", "0", "π"])
    else:
        ax.set_xticks([-2*np.pi, 0, 2*np.pi], ["-2π", "0", "2π"])

    return None

def add_symmetric_angle(array):
    """
    Function for plotting. Computations get results for, e.g., [-2, -1, 0, 1, 2, 3], so we want to have
    also the [-3], which should be identical to 3. We add this missing value to the passed array.
    """
    return np.insert(array, [0], array[-1])