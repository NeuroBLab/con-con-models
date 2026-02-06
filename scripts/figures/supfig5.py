import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import pandas as pd

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.currents as curr
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.utils as dutl


import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils

import ccmodels.utils.angleutils as au


def compute_conn_prob(v1_neurons, v1_connections, proofread):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, proofread=proofread)
    meandata  = {}
    errordata = {}
    n_connections = {}

    post_ids = fl.filter_neurons(v1_neurons, tuning='tuned', layer="L23", proofread=proofread[1])['id'].unique()

    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] #/p.loc[0, "mean"]
        meandata[layer]  = p['mean'].values 
        errordata[layer]  = p['std'].values 

        pre_ids  = fl.filter_neurons(v1_neurons, tuning='tuned', layer=layer, proofread=proofread[0])['id'].unique()
        n_connections[layer] = len(fl.synapses_by_id(v1_connections, pre_ids=pre_ids, post_ids=post_ids, who='both'))

    return meandata, errordata, n_connections

def conn_prob_osi(ax, probmean, proberr, n_connections, layer, label, color):

    #Plot it!
    angles = np.linspace(0, np.pi/2, 5)

    low_band  = probmean[layer] - proberr[layer]
    high_band = probmean[layer] + proberr[layer]

    ax.fill_between(angles, low_band, high_band, color = color, alpha = 0.2)
    ax.plot(angles, probmean[layer], color = color, label = label.replace("->", "→") + rf", $N_\text{{syn}}=${n_connections[layer]}")
    ax.scatter(angles, probmean[layer], color=color, s=cr.ms, zorder = 3)

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_xticks([0, np.pi/4, np.pi/2], ["0", "π/4", "π/2"])
    ax.set_ylim(-0.005, 0.1)

def conn_prob_osi_norm(ax, probmean, proberr, norm, layer, label, color):
    angles = np.linspace(0, np.pi/2, 5)

    low_band  = (probmean[layer] - proberr[layer]) / norm
    high_band = (probmean[layer] + proberr[layer]) / norm

    ax.fill_between(angles, low_band, high_band, color = color, alpha = 0.2)
    ax.plot(angles, probmean[layer] / norm, color = color)
    ax.scatter(angles, probmean[layer] / norm, color=color, s=cr.ms, zorder = 3)

    ax.tick_params(axis='both', which='major')
    ax.set_xticks([0, np.pi/4, np.pi/2], ["0", "π/4", "π/2"])
    ax.set_ylim(0., 0.5)

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname):
    # load files
    units, connections, rates, rates_err = loader.load_data(return_error=True)
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()


    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)



    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.4), layout='constrained')

    axes = fig.subplot_mosaic(
        [["T", "T"],
        ["L23l", 'L4l'],
        ["L23b", 'L4b']],
        height_ratios=[0.4, 1, 1.]
    )

    axes["T"].set_axis_off()
    axes['T'].text(0.2, 1., 'Layer 2/3',    weight='bold', ha='center')
    axes['T'].text(0.75, 1., 'Layer 4',      weight='bold', ha='center')


    proofreadcases = {"extended -> all" : ["ax_extended", None], "clean -> all":["minimum", None], "extended -> extended":["ax_extended", "dn_extended"]}

    colors = {} 

    for layer in ['L23', 'L4']:
        colorshades = cr.darken(cr.lcolor[layer], 2, 0.35) 
        colors["clean -> all"] = cr.lcolor[layer] 
        colors["extended -> all"] = colorshades[0] 
        colors["extended -> extended"] = colorshades[1] 

        for pcase in ['clean -> all', 'extended -> all', 'extended -> extended']:
            probmean, proberr, n_connections = compute_conn_prob(matched_neurons, matched_connections, proofread=proofreadcases[pcase])
            conn_prob_osi(axes[f"{layer}l"], probmean, proberr, n_connections, layer, pcase, colors[pcase])
            conn_prob_osi_norm(axes[f"{layer}b"], probmean, proberr, probmean[layer].sum(), layer, pcase, colors[pcase])

        #axes[f"{layer}l"].legend(loc=(0.2, 0.8), ncols=1, fontsize=9)
        axes[f'{layer}b'].set_xlabel(r"$|\hat \theta_\text{post} - \hat \theta _\text{pre}|$")

    #Add both legends to the text axis. using XX.legend only one can be, so assign to variable and readd using add_artist
    handles1, labels1 = axes[f"L23l"].get_legend_handles_labels()
    leg1 = axes["T"].legend(handles1, labels1, loc=(0.05, 0.),fontsize=9, bbox_transform=axes["T"].transAxes)
    handles2, labels2 = axes[f"L4l"].get_legend_handles_labels()
    leg2 = axes["T"].legend(handles2, labels2, loc=(0.6, 0.),fontsize=9, bbox_transform=axes["T"].transAxes)
    axes["T"].add_artist(leg1)

    axes['L23l'].set_ylabel("Conn. Prob\n")
    axes['L23b'].set_ylabel("Conn. Prob\n(Normalized)")

    axes2label = [axes[k] for k in ['L23l', 'L4l', 'L23b', 'L4b']]
    label_pos  = [[0.02, 0.95]] * 4 
    sty.label_axes(axes2label, label_pos)
    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure("supfig_pcon.pdf")
