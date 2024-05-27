import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import argparse
from tqdm.auto import tqdm
from PIL import Image
import sys
sys.path.append(".")
import cmasher as cmr
import ccmodels.analysis.functions as ccf
import ccmodels.plotting.utils as ccu
import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.analysis.functions as funcs


def load_sim_data(path2data):
    """
    Load the data from the simulations
    """

    #This should coincide with the simulations, here I set it by hand
    #TODO improve this to avoid leaving number flying everywhere
    OSI_model = np.empty((100, 10))
    J_values = np.logspace(-1, 1, 100)
    g_values = np.linspace(1.5, 7.5, 10)

    for i in range(100):
        for j in range(10):
            try:
                re_model, ri, rx = funcs.opendata(i,j, path=path2data, returnact=False) 
                Theta = np.arange(0, 2*np.pi, np.pi/8.)
                OSI_E = funcs.compute_orientation_selectivity_index(re_model, Theta)    
                OSI_model[i,j] = OSI_E.mean() 
            except:
                OSI_model[i,j] = 0.0
 
    return OSI_model, J_values, g_values

def load_exp_data():
    Theta = np.arange(0, 2*np.pi, np.pi/8.)
    v1_neurons = pd.read_pickle('../con-con-models/data/v1l234_neurons.pkl')
    expneurons = np.empty((len(v1_neurons["activity"]), 16))
    for i,act in enumerate(v1_neurons["activity"].values):
        expneurons[i,:] = act

    OSI_exp = funcs.compute_orientation_selectivity_index(expneurons, Theta)     
    return OSI_exp, expneurons.mean(axis=0) 

def get_best_fits(OSI_model, OSI_exp, J_values):
    #For each value of g, get which ones is the best J. 
    #Store these values for plotting
    best_per_g = np.empty(10)
    best_per_g_ix = np.empty(10, dtype=int)
    for j in range(10):
        idx  = np.abs(OSI_exp - OSI_model[:,j]).argmin()
        best_per_g[j] = J_values[idx]
        best_per_g_ix[j] = int(idx) 

    #Get absolute best
    idx_best = np.abs(OSI_exp - OSI_model).argmin()
    idx_best = np.unravel_index(idx_best, OSI_model.shape)

    return best_per_g, best_per_g_ix, idx_best


def plot_imshow(ax, OSI_model, best_per_g, g_values):
    #Plot the 2D phase diagram
    im = ax.imshow(OSI_model.transpose(), vmin=0, vmax=1, extent=[0.1, 8, 1.5, 7.5], aspect="auto", cmap=cmr.ember)

    #Points of best fit for each J
    ax.scatter(best_per_g, g_values, color="white")

    #Label properties
    ax.set_xscale("log")
    ax.set_xlim(0.1, 8)

    ax.set_xlabel("J")
    ax.set_xlabel("g")

    #Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)


def plot_lines(ax, J_values, OSI_model, OSI_exp_mean, g_idx, g_values, best_per_g, best_per_g_ix, color=["green", "red"]):
    #Some cuts of the diagram
    for gix, c in zip(g_idx, color):
        osi = OSI_model[:, gix]
        ax.plot(J_values[osi > 0], osi[osi > 0], label=f"g = {g_values[gix]:.2f}", color=c)

        Jbest = best_per_g[gix]
        ax.plot([Jbest, Jbest], [-1, osi[best_per_g_ix[gix]]], color="gray", ls=":")
    
    ax.axhline(OSI_exp_mean, ls=":", color="gray")
    ax.set_xscale("log")

    ax.set_xlabel("J")
    ax.set_xlabel("OSI")

    ax.set_xticks([1e-1, 1e0, Jbest], labels=[1e-1, 1e0, r"$\mathregular{J_b}$"])

    ax.set_xlim(0.1, 7)
    ax.set_ylim(-0.01, 1)
    ax.legend(loc = (0.1, 0.7))


def compare_OSI(ax, keys_to_plot, OSI_dict, labels_dict, colors_dict, ls_dict, nbins=50, ymax=2.5):
    Theta = np.arange(0, 2*np.pi, np.pi/8.)
    bins = np.linspace(0, 1, nbins)

    for key in keys_to_plot:
        OSI   = OSI_dict[key]
        label = labels_dict[key]
        c     = colors_dict[key]
        ls    = ls_dict[key]

        ax.hist(OSI, bins, histtype='step', color=c, density=True, label=label)

        ax.axvline(OSI.mean(), color=c, ls=ls, ymax=0.9)

    ax.set_xlabel("OSI")
    ax.set_ylabel("p(OSI)")

    ax.legend(loc=(0.05, 0.9), ncols=3)
    ax.set_ylim(0, ymax)

def compare_rates(ax, re_model, re_exp): 
    angles = np.arange(-np.pi, np.pi, np.pi/8.)
    #Set for how much we will have these 0 at left and right
    offset = angles[2] - angles[0]
    left, right = angles[0] - offset, angles[-1] + offset
    angles = np.insert(angles, [0, angles.size], [left, right])

    re_model = np.insert(re_model, [0, re_model.size], [0, 0])
    re_exp   = np.insert(re_exp, [0, re_exp.size], [0, 0])

    #Plot
    ax.step(angles, re_model, where="mid", color="green", label="Model")
    ax.step(angles, re_exp, where="mid", color="blue", label="Experiment")

    ax.set_xticks([ -np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-π', r'- π/2', '0', r'π/2', r'π'])
    
    ax.set_xlabel("θ")
    ax.set_ylabel("Average Rate")
    
    ax.legend(loc=(0.7, 0.8))

# ----------------------------------------------------------------------------------------

# ======================================================
# --------------- FIGURE STRUCTURE ---------------------
# THis is the code that loads the data, structures the 
# location of the panels, and then call the analysis 
# functions to fill in the panels, via the functions above.
# ======================================================

def plot_figure():
    #Defining Parser
    parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

    #Adding and parsing arguments
    parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
    args = parser.parse_args()


    OSI = {}

    #Load data and get the best fits
    OSI["model"], J_values, g_values = load_sim_data("ccmodels/victor/proteus2")
    OSI["exp"], re_exp = load_exp_data()
    best_per_g, best_per_g_ix, idx_best = get_best_fits(OSI["model"], OSI["exp"].mean(), J_values)

    best = (J_values[idx_best[0]], g_values[idx_best[1]])
    print("Best fit = ", best)
    re_best, ri, rx = funcs.opendata(idx_best[0], idx_best[1], path="ccmodels/victor/proteus2", returnact=False)
    Theta = np.arange(0, 2*np.pi, np.pi/8.)
    OSI["best"] = funcs.compute_orientation_selectivity_index(re_best ,Theta)    
    del ri, rx

    re_other, ri, rx = funcs.opendata(idx_best[0], 1, path="ccmodels/victor/proteus2", returnact=False)
    OSI["best_glow"] = funcs.compute_orientation_selectivity_index(re_other, Theta)    
    del ri, rx

    ix_low = 50
    re, ri, rx = funcs.opendata(ix_low, 7, path="ccmodels/victor/proteus2", returnact=False)
    OSI["Jlow"]  = funcs.compute_orientation_selectivity_index(re, Theta)    
    del re, ri, rx

    ix_high = 80
    re, ri, rx = funcs.opendata(ix_high, 7, path="ccmodels/victor/proteus2", returnact=False)
    OSI["Jhigh"] = funcs.compute_orientation_selectivity_index(re, Theta)    
    del re, ri, rx

    sty.master_format()


    fig, axes = plt.subplot_mosaic(
        """
        AB
        CD
        """, 
        figsize=sty.two_col_size(ratio=1.5), layout="constrained", gridspec_kw={"height_ratios":[1,1]})



    plot_imshow(axes["A"], OSI["model"], best_per_g, g_values)
    plot_lines(axes["C"], J_values, OSI["model"], OSI["exp"].mean(), [1, 7], g_values, best_per_g, best_per_g_ix)


    # -----------------

    labels = {}
    labels["best"]      = f"g = {g_values[idx_best[1]]:.2f}"
    labels["best_glow"] = f"g = {g_values[1]:.2f}"
    labels["exp"]       = f"Experiment"
    labels["Jlow"]      = f"J = {J_values[ix_low]:.2f}"
    labels["Jhigh"]      = f"J = {J_values[ix_high]:.2f}"

    colors = {}
    colors["best"]      = "red" 
    colors["best_glow"] = "green" 
    colors["exp"]       = "blue" 
    colors["Jlow"]      = "teal" 
    colors["Jhigh"]     = "mediumvioletred" 

    ls = {}
    ls["best"]      = "-." 
    ls["best_glow"] = ":" 
    ls["exp"]       = ":" 
    ls["Jlow"]      = ":" 
    ls["Jhigh"]     = ":" 
    #compare_OSI(axes["C"], OSI["best"], , [best[1], g_values[1]], OSI_exp, nbins=40)
    compare_OSI(axes["B"], ["best", "best_glow", "exp"], OSI, labels, colors, ls) 



    OSI["original"] = np.load("ccmodels/victor/osie_original.npy")
    OSI["reshuffled"] = np.load("ccmodels/victor/osie_reshuffled.npy")
    ls["original"]      = ":" 
    ls["reshuffled"]     = ":" 
    colors["original"]      = "red" 
    colors["reshuffled"]     = "fuchsia" 
    labels["original"] = "Best"
    labels["reshuffled"] = "Exc Reshuffle"

    compare_OSI(axes["D"], ["original", "reshuffled"], OSI, labels, colors, ls)

    #compare_rates(axes["D"], re_best.mean(axis=0), re_exp)


    fig.savefig(args.save_destination+"fig4b.pdf", bbox_inches="tight")