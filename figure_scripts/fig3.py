import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm.auto import tqdm
from PIL import Image
import sys
sys.path.append(".")

import ccmodels.dataanalysis.currents as dcr
import ccmodels.dataanalysis.utils as utl
import ccmodels.dataanalysis.dfs2numpy as d2n



import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils

def plot_input_current(ax, angles, currents, half=True):
    norm_c = currents["Total"]["norm"]

    #Normalization constants
    for layer in ["Total", "L2/3", "L4"]:
        av_cur = currents[layer]["av_curr"] / norm_c
        yerror = currents[layer]["std_curr"] / norm_c

        ax.fill_between(angles, utl.shift_4_plot(av_cur - yerror), utl.shift_4_plot(av_cur + yerror), color=cr.lcolor[layer], alpha=0.2)

        ax.plot(angles, utl.shift_4_plot(av_cur), lw = 1, color = cr.lcolor[layer], zorder = 2, label = layer)
        ax.scatter(angles, utl.shift_4_plot(av_cur), color = 'black', s = 5, zorder = 3)

    plotutils.get_xticks(ax, half=half)

    ax.set_xlabel("Δθ")
    ax.set_ylabel("Avg. Current")

def plot_single_current(ax, angles, inputs, nangles=16, half=True):

    for r in inputs:
        rangle = utl.shift_4_plot(r)
        ax.plot(angles, rangle,  lw=1)
        ax.scatter(angles, rangle, color="black", s=5, zorder=3)

    ax.set_xlabel("Δθ")
    ax.set_ylabel("Single Synp. Current")
    plotutils.get_xticks(ax, half=half)

def plot_distrib_diffrates(ax, diffs):

    nbins = 70 
    bins = np.linspace(-10, 10, nbins)

    total_diffs = [] 

    for layer in ["L2/3", "L4"]:
        total_diffs += diffs[layer]
        ax.hist(diffs[layer], bins, density=True, cumulative=False, histtype='step', lw=2, color = cr.lcolor[layer])
        #ax.axvline(np.mean(diffs[layer]), color=cr.lcolor[layer], ls=":")



    ax.hist(total_diffs, bins, density=True, cumulative=False, histtype='step', lw=2, color =cr.lcolor["Total"])

    ax.set_yscale("log")
    ax.set_xlabel("Δ")
    ax.set_ylabel("P(Δ)")

def plot_bootstraps(ax, angles, bootstrap, half=True):

    for layer in ["L2/3", "L4"]:
        #angles = bootstrap[layer].index.values
        current = bootstrap[layer]["shifted_current"]
        ax.plot(angles, current, color=cr.lcolor[layer])
    
    total_activity = np.array(bootstrap["L2/3"]["shifted_current"]) + np.array(bootstrap["L4"]["shifted_current"])  
    ax.plot(angles, total_activity, color=cr.lcolor["Total"])

    plotutils.get_xticks(ax, half=half)
    
    ax.set_xlabel("Δθ")
    ax.set_ylabel("Bootstrap Current")

#TODO check the step plot
def plot_dist_bootstrap(ax, angles, prob_pref_ori, color=cr.lcolor["Total"], label=""): 


    #Add a zero at beginning and end for a more beatiful plot
    #prob_pref_ori = np.insert(prob_pref_ori, [0, prob_pref_ori.size], [0, 0])

    #Set for how much we will have these 0 at left and right
    #offset = angles[2] - angles[0]
    #left, right = angles[0] - offset, angles[-1] + offset
    #angles = np.insert(angles, [0, angles.size], [left, right])

    #Plot
    ax.plot(angles, utl.shift_4_plot(prob_pref_ori), color=color, label=label)
    #ax.step(angles, prob_pref_ori, where="mid", color=color, label=label)
    plotutils.get_xticks(ax, max=np.pi, half=True)

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
    ax.axvline(0.0, color="black")
    ax.axvline(1.57, color="black", ls=":")


#Read and process necessary data
orientation_only = True
v1_neurons, v1_connections, rates = utl.load_data(half_angle=orientation_only)
vij = d2n.get_adjacency_matrix(v1_neurons, v1_connections)
angles = plotutils.get_angles(kind="centered", half=orientation_only)

# --------------------

#TODO changing this rn 
#avg_input_l23, avg_input_l4, fraction_l3, fraction_l4 = dcr.compute_inpt_curr_by_layer(v1_neurons, vij, rates) 
#plot_input_current(axes["A"], angles, avg_input_l23, fraction_l3, avg_input_l4, fraction_l4)
currents = dcr.compute_inpt_curr_by_layer(v1_neurons, vij, rates)
plot_input_current(axes["A"], angles, currents)
axes["A"].legend(loc=(0.1, 0.25))


# ------------

n_neurons = 3
inputs_single = [dcr.single_synapse_current(v1_neurons, v1_connections, vij, rates) for i in range(n_neurons)]
plot_single_current(axes["B"], angles, inputs_single)
for angle in [np.pi/2, -np.pi/2]:
    axes["B"].axvline(angle, color="black", ls=":")

# ------------

diffrate = dcr.compute_distrib_diffrate_allsynapses(v1_neurons, v1_connections, vij, rates)
plot_distrib_diffrates(axes["C"], diffrate)

# ------------

tuned_outputs = utl.filter_connections(v1_neurons, v1_connections, tuned=True, who="pre") 

nexperiments = 1000

prob_pref_ori = dcr.compute_inpt_bootstrap(v1_neurons, tuned_outputs, nexperiments, rates)
plot_dist_bootstrap(axes["D"],  angles, prob_pref_ori, label="Sampled Current")

#prob_pref_ori = dcr.compute_inpt_bootstrap(v1_neurons, tuned_outputs, nexperiments, rates, shifted=True)
#plot_dist_bootstrap(axes["D"],  angles, prob_pref_ori, color="red", label="Shuffled Current")

#axes["D"].set_ylim(0, 0.08)
axes["D"].legend(loc=(0.1, 0.9))

fig.savefig(args.save_destination+"fig3.pdf", bbox_inches="tight")