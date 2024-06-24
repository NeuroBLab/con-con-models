import cmasher as cmr
from KDEpy import FFTKDE

import numpy as np
import matplotlib.pyplot as plt

def plot_posterior_distrib(axes, posterior_samples, intervals, inferred, cmap=None, lw=2.0, bw='ISJ'):
    """
    Plot a graph representing the posterior distribution for each parameter.
    Parameters
    ==========
    - axes : matploblib axes ndarray
        An array containing the axes where to plot it.
    - posterior_samples : ndarray
        The array containing the set of samples to be histogrammed, for each parameter. NparxNdata
    - intervals : ndarray
        Contains the [xmin, xmax] for each one of the parameters. 
    - correct_pars, inferred : ndarray
        The arrays containing ground truth and the inferred parameters, respectively
    - fitting_eqs : string
        Fitting model we are currently employing. Sets x labels. 
    """

    #Plot with a different colormap to differentiate from rel error plots
    if cmap is None:
        cmap = cmr.amber
    color = cmap(0.6)
    colorline = cmap(0.3)

    labels = ["J", "g", "θ", "σ", "τI"]

    #A plot for each parameter
    for param in range(axes.size):

        #Histogram
        bins = np.linspace(intervals[param][0], intervals[param][1], 100)
        hist, edges = np.histogram(posterior_samples[:,param], bins=bins, density=True)
        centered = 0.5*(edges[1:]+edges[:-1])
        
        x, y = FFTKDE(kernel='gaussian', bw=bw).fit(posterior_samples[:,param].numpy()).evaluate()
        #Fill between for fancyness
        #axes[param].fill_between(centered, np.zeros(99), hist, color=color, lw=2.0, alpha=0.5)
        axes[param].fill_between(x, np.zeros(len(x)), y, color=color, lw=2.0, alpha=0.5)

        #Now highlight correct and most common (estimation)
        axes[param].axvline(inferred[param], c=colorline, ls="--", lw=lw)
        
        
        #Despine and clean axes
        axes[param].spines['right'].set_visible(False)
        axes[param].spines['top'].set_visible(False)
        axes[param].set_yticks([])
        axes[param].set_ylim(0, 1.1*np.max(hist))

        #Set labels
        axes[param].set_xlabel(labels[param], fontsize=14)


    #Finish graph
    axes[0].set_ylabel("Prob. density", fontsize=14)
    return



def plot_posterior_correlations(axes, posterior_samples, intervals):
    """
    Plot a graph representing the posterior distribution for each parameter.
    Parameters
    ==========
    - axes : matploblib axes ndarray
        An array containing the axes where to plot it.
    - posterior_samples : ndarray
        The array containing the set of samples to be histogrammed, for each parameter. NparxNdata
    - intervals : ndarray
        Contains the [xmin, xmax] for each one of the parameters. 
    - correct_pars, inferred : ndarray
        The arrays containing ground truth and the inferred parameters, respectively
    """


    labels = ["J", "g", "θ", "σ"]
    pairs = [[0,1], [2,3], [0,2], [0,3]]

    for index, ap in enumerate(pairs):
        
        i,j = index//2, index%2

        xbins = np.linspace(intervals[ap[0]][0], intervals[ap[0]][1], 100)
        ybins = np.linspace(intervals[ap[1]][0], intervals[ap[1]][1], 100)
        hist, xedg, yedg = np.histogram2d(posterior_samples[:,ap[0]].numpy(), posterior_samples[:,ap[1]].numpy(), bins=(xbins, ybins), density=True) 

        axes[i,j].imshow(hist.transpose(), interpolation="gaussian", origin="lower", cmap=cmr.amber, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect="auto")
        axes[i,j].set_xlabel(labels[ap[0]], fontsize=14)
        axes[i,j].set_ylabel(labels[ap[1]], fontsize=14)