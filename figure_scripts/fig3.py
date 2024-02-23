import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm.auto import tqdm
from PIL import Image
import sys
sys.path.append(".")
from ccmodels.plotting.utils import compute_avg_inpt_current, single_synapse_current, compute_distrib_diffrate_allsynapses, compute_inpt_bootstrap, prepare_c3, prepare_d3, prepare_e3
import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr

def plot_input_current(ax, avg_input_l23, fraction_l3, avg_input_l4, fraction_l4, dir_range="full"):
    norm_c = np.max((avg_input_l23['avg_cur']*fraction_l3) + (avg_input_l4['avg_cur']*fraction_l4))

    #Normalization constants
    current = (avg_input_l23['avg_cur']*fraction_l3) + (avg_input_l4['avg_cur']*fraction_l4)
    yerror_t =np.sqrt((avg_input_l23['cur_sem']**2)+(avg_input_l4['cur_sem']**2))

    #Total Current
    ax.fill_between(avg_input_l23['dirs'], (current- yerror_t)/norm_c, (current + yerror_t)/norm_c, color=cr.lcolor["Total"], alpha=0.2)
    ax.plot(avg_input_l23['dirs'], current/norm_c, lw = 1, color = cr.lcolor["Total"], zorder = 2, label = 'Total current')
    ax.scatter(avg_input_l23['dirs'], current/norm_c, color = 'black', s = 5, zorder = 3)


    #L2/3
    ax.fill_between(avg_input_l23['dirs'], ((avg_input_l23['avg_cur']*fraction_l3) - yerror_t)/norm_c, ((avg_input_l23['avg_cur']*fraction_l3) + yerror_t)/norm_c, color=cr.lcolor["L2/3"], alpha=0.2)

    ax.plot(avg_input_l23['dirs'], avg_input_l23['avg_cur']*fraction_l3/norm_c, lw= 1, color = cr.lcolor["L2/3"], label = 'L2/3 current')
    ax.scatter(avg_input_l23['dirs'], avg_input_l23['avg_cur']*fraction_l3/norm_c, color = 'black', s = 5, zorder=3)


    #L4
    ax.fill_between(avg_input_l23['dirs'], ((avg_input_l4['avg_cur']*fraction_l4)- yerror_t)/norm_c, ((avg_input_l4['avg_cur']*fraction_l4) + yerror_t)/norm_c, color=cr.lcolor["L4"], alpha=0.2)

    ax.plot(avg_input_l4['dirs'], avg_input_l4['avg_cur']*fraction_l4/norm_c, lw= 1, color = cr.lcolor["L4"], label =' L4 current')
    ax.scatter(avg_input_l4['dirs'], avg_input_l4['avg_cur']*fraction_l4/norm_c, color = 'black',s = 5, zorder=3)


    if dir_range=="full":
        ax.set_xticks([ -np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-π', r'- π/2', '0', r'π/2', r'π'])
    else:
        ax.set_xticks([0, np.pi/2, np.pi], ['0', r'π/2', r'π'])

    ax.set_xlabel("Δθ")
    ax.set_ylabel("Avg. Current")


def plot_single_current(ax, inputs, dir_range="full"):

    for layer in ["L2/3", "L4"]:
        angles = inputs[layer]["new_dirs"].values[0]
        currnt = inputs[layer]["shifted_current"].values[0]
        ax.plot(angles, currnt, color=cr.lcolor[layer], lw=1)
        ax.scatter(angles, currnt, color="black", s=5, zorder=3)

    ax.set_xlabel("Δθ")
    ax.set_ylabel("Single Synp. Current")


    if dir_range=="full":
        ax.set_xticks([ -np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-π', r'- π/2', '0', r'π/2', r'π'])
    else:
        ax.set_xticks([0, np.pi/2, np.pi], ['0', r'π/2', r'π'])

def plot_distrib_diffrates(ax, diffs, dir_range="full"):

    nbins = 70 
    #bins = np.linspace(0, 30, nbins)
    bins = np.logspace(0, 3, nbins)

    total_diffs = [] 

    for layer in ["L2/3", "L4"]:
        total_diffs += diffs[layer]
        ax.hist(np.abs(diffs[layer]), bins, density=True, cumulative=False, histtype='step', lw=2, color = cr.lcolor[layer])
        #ax.axvline(np.mean(diffs[layer]), color=cr.lcolor[layer], ls=":")



    ax.hist(np.abs(total_diffs), bins, density=True, cumulative=False, histtype='step', lw=2, color =cr.lcolor["Total"])

    meantotal = np.mean(total_diffs)
    stdtotal = np.std(total_diffs)
    #ax.axvline(meantotal, color=cr.lcolor["Total"], ls=":")
    #ax.axvspan(meantotal - stdtotal, meantotal + stdtotal, color=cr.lcolor["Total"], ls=":", alpha=0.1)
    #ax.text(0.4, 0.8, "68% CI", color=cr.lcolor["Total"])

    ax.set_xlim(1, 100)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Δ")
    ax.set_ylabel("P(Δ)")

def plot_bootstraps(ax, bootstrap, dir_range="full"):

    for layer in ["L2/3", "L4"]:
        angles = bootstrap[layer].index.values
        current = bootstrap[layer]["shifted_current"]
        ax.plot(angles, current, color=cr.lcolor[layer])
    
    total_activity = np.array(bootstrap["L2/3"]["shifted_current"]) + np.array(bootstrap["L4"]["shifted_current"])  
    ax.plot(angles, total_activity, color=cr.lcolor["Total"])

    if dir_range=="full":
        ax.set_xticks([ -np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-π', r'- π/2', '0', r'π/2', r'π'])
    else:
        ax.set_xticks([0, np.pi/2, np.pi], ['0', r'π/2', r'π'])
    
    ax.set_xlabel("Δθ")
    ax.set_ylabel("Bootstrap Current")

def plot_dist_bootstrap(ax, angles, bootstrap, dir_range="full"):

    if dir_range=="half":
        angles = angles[4:]
        bootstrap = np.array(bootstrap)
        bootstrap[bootstrap < 0.] = np.pi + bootstrap[bootstrap < 0.] 

    ax.hist(bootstrap, angles, density=True, cumulative=False, histtype='step', lw=2, color =cr.lcolor["Total"])

    if dir_range=="full":
        ax.set_xticks([ -np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-π', r'- π/2', '0', r'π/2', r'π'])
    else:
        ax.set_xticks([0, np.pi/2, np.pi], ['0', r'π/2', r'π'])
    
    ax.set_xlabel("θ preferred")
    ax.set_ylabel("P(θ)")

# ----------------------------------------------------------------------------------------

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

#Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()


sty.master_format()


fig, axes = plt.subplot_mosaic(
    """
    AB
    CD
    """, 
    figsize=sty.two_col_size(ratio=1.5), layout="constrained", gridspec_kw={"height_ratios":[1,1]})

for k in "AB":
    ax = axes[k]
    ax.spines["left"].set_visible(False)
    ax.axvline(0.0, color="black")
    ax.axvline(1.57, color="black", ls=":")


#Read and process necessary data
v1_neurons = pd.read_pickle('../con-con-models/data/v1l234_neurons.pkl')
v1_connections = pd.read_pickle('../con-con-models/data/v1l234_connections.pkl')
proofread_input_n = pd.read_csv('../con-con-models/data/proofread_l234_inputs.csv')


# Angles from -pi to pi or just 0 to pi?
dir_range = "full"

# --------------------

avg_input_l23, fraction_from_l23, avg_input_l4, fraction_from_l4 = compute_avg_inpt_current(v1_connections, proofread_input_n, dir_range)
plot_input_current(axes["A"], avg_input_l23, fraction_from_l23, avg_input_l4, fraction_from_l4, dir_range=dir_range)


# ------------

n_neurons = 1
inputs_single = single_synapse_current(v1_connections, n_neurons, dir_range=dir_range, seed=4792)
plot_single_current(axes["B"], inputs_single, dir_range=dir_range)

# ------------

diffrate = compute_distrib_diffrate_allsynapses(v1_connections, dir_range=dir_range)
plot_distrib_diffrates(axes["C"], diffrate, dir_range=dir_range)

# ------------

#input_bootstrap = compute_inpt_bootstrap(v1_connections, 1000, dir_range=dir_range)
#plot_bootstraps(axes["D"], input_bootstrap, dir_range=dir_range)

angles, preferred_oris = compute_inpt_bootstrap(v1_connections, 100, dir_range=dir_range)
plot_dist_bootstrap(axes["D"], angles, preferred_oris, dir_range="full")



fig.savefig(args.save_destination+"fig3.pdf", bbox_inches="tight")