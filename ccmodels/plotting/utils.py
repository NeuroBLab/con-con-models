import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats

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
        #return np.linspace(-maxangle/2, maxangle/2, ndivisions+1)[1:]
        return np.linspace(-maxangle/2, maxangle/2, ndivisions+1)
    elif kind=="diff":
        return np.linspace(-maxangle, maxangle, ndivisions + 1)

def add_symmetric_angle(array):
    """
    Function for plotting. Computations get results for, e.g., [-2, -1, 0, 1, 2, 3], so we want to have
    also the [-3], which should be identical to 3. We add this missing value to the passed array.
    """
    return np.insert(array, [0], array[-1])


def shift(observable):
    """
    This function changes an observable measured in [0, 2pi[ to be plotted in [-pi, pi] (including both extreme) 
    for nice-looking plots.
    """

    return add_symmetric_angle(np.roll(observable, len(observable)//2-1))

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

# ------------------------------------------------------------
# ----------------------------- TESTS ------------------------
# ------------------------------------------------------------

#TODO can be improved so we don't have to manually set the size of the whiskers
def test_compare(ax, stats_a, stats_b, pos_a, pos_b, ybase, yfrac=0.03, yoffset = 0.05):
    result = ttest_ind_from_stats(mean1=stats_a["mean"], std1=stats_a["std"], nobs1=stats_a["size"],
                                  mean2=stats_b["mean"], std2=stats_b["std"], nobs2=stats_b["size"], 
                                  alternative="greater")

    ymax = ybase + yfrac 
    ax.plot([pos_a, pos_a, pos_b, pos_b], [ymax, ybase, ybase, ymax], color="black")

    midpoint = 0.5*(pos_a+pos_b) 

    if result.pvalue > 0.05:
        result = "n.s."
        text_offset = 0.01 * midpoint
    elif result.pvalue > 0.01:
        result = "*"
        text_offset = 0.01 * midpoint
    elif result.pvalue > 0.001:
        result = "* *"
        text_offset = 0.01 * midpoint
    else:
        result = "* * *" 
        text_offset = 0.015 * midpoint


    ytext = ybase - yoffset 
    ax.text(midpoint - text_offset, ytext, result)

    return