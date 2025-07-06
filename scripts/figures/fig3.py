import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from PIL import Image

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

def show_image(ax, path2im):
    im = Image.open("images/" + path2im)
    ax.set_axis_off()
    ax.imshow(im)

def plot_sampling_current(ax, ax_normalized, v1_neurons, v1_connections, rates, indegree, nexperiments=1000):
    angles = plotutils.get_angles(kind="centered", half=True)

    #Compute the currents in the system
    #mean_cur = curr.bootstrap_system_currents_shuffle(v1_neurons, v1_connections, rates, nexperiments, frac=frac)
    mean_cur, std_cur = curr.bootstrap_mean_current(indegree, v1_neurons, v1_connections, rates, nexperiments)

    #Total current is shown just in the "unnormalized" version. Also we need to obtain
    #the global total current to normalize according to it
    #total_cur = mean_cur['Total'].mean(axis=1)
    #norma = np.max(total_cur)
    total_cur = plotutils.shift(mean_cur["Total"])
    ax.plot(angles, total_cur, label='Total', color=cr.lcolor['Total'])
    ax.scatter(angles, total_cur, color=cr.dotcolor['Total'], s=cr.ms, zorder=3)

    #Then show L23 and L4 currents for unnormalized and normalized versions
    for layer in ['L23', 'L4']:
        meancur = plotutils.shift(mean_cur[layer])
        stdcur = plotutils.shift(std_cur[layer])
        
        ax.fill_between(angles, (meancur-stdcur), (meancur+stdcur), color=cr.lcolor[layer], alpha=0.2)
        ax.plot(angles, meancur, label=layer, color=cr.lcolor[layer])
        ax.scatter(angles, meancur, color=cr.dotcolor[layer], s=cr.ms, zorder=3)

        stdcur  /= np.max(meancur)
        meancur /= np.max(meancur)
        ax_normalized.fill_between(angles, meancur-stdcur, meancur+stdcur, color=cr.lcolor[layer], alpha=0.2)
        ax_normalized.plot(angles, meancur, label=layer, color=cr.lcolor[layer])
        ax_normalized.scatter(angles, meancur, color=cr.dotcolor[layer], s=cr.ms, zorder=3)

    plotutils.get_xticks(ax, max=np.pi, half=True)
    plotutils.get_xticks(ax_normalized, max=np.pi, half=True)

    ax.set_xlabel(r'$\hat \theta_\text{post} - \theta$')
    ax_normalized.set_xlabel(r'$\hat \theta_\text{post} - \theta$')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('μ(Δθ)')
    ax_normalized.set_ylabel('μ(Δθ)/μ(0)')

    
def plot_sampling_current_peaks(ax, v1_neurons, v1_connections, rates, indegree):

    frac = indegree / len(v1_connections)

    current = curr.bootstrap_system_currents_peaks(v1_neurons, v1_connections, rates, frac=frac, nexperiments=1000)
    bins = np.arange(-7.5, 8.5, 1)

    for layer in ['L23', 'L4']:
        pref_ori = np.argmax(current[layer], axis=1)
        pref_ori[pref_ori > 3] = pref_ori[pref_ori > 3] - 8 

        hist, _ = np.histogram(pref_ori, bins=bins)
        ax.step(bins[1:], hist, color = cr.lcolor[layer], label=layer)


def tuning_prediction_performance(ax, matched_neurons, matched_connections, rates, indegree, nexperiments=1000): 

    angles = np.arange(9)
    tuned_outputs = fl.filter_connections(matched_neurons, matched_connections, tuning="matched", who="post") 

    prob_pref_ori  = curr.sample_prefori(matched_neurons, tuned_outputs, nexperiments, rates, nsamples=indegree)
    

    #Plot
    for layer in ['Total', 'L23', 'L4']:
        ax.plot(angles, plotutils.shift(prob_pref_ori[layer]), color=cr.lcolor[layer], label=layer)
        ax.scatter(angles, plotutils.shift(prob_pref_ori[layer]), color=cr.dotcolor[layer], zorder=3, s=cr.ms) 


    ax.set_xlabel(r"$\hat \theta_\text{target} - \hat \theta_\text{emerg}$")
    ax.set_ylabel("Fract. correct")

    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_yticks([0, 0.25, 0.5])



#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname):
    # load files
    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)

    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.9), layout="constrained")

    axes = fig.subplot_mosaic(
        """
        XXX
        ABC
        """
    )

    #Given an exc -> exc in-degree kee, how many inputs does a neuron receive in total?
    #Compute it using the probabilities obtained from the data 
    kee = 150
    conn_prob = pd.read_csv("data/model/prob_connectomics_cleanaxons.csv", index_col=0)
    indegree = int(kee * (1.0 + conn_prob.loc['E', 'X'] / conn_prob.loc['E', 'E']))

    show_image(axes["X"], "sketchsampling.png")

    plot_sampling_current(axes["A"], axes["B"], matched_neurons, matched_connections, rates, indegree)
    tuning_prediction_performance(axes['C'], matched_neurons, matched_connections, rates, indegree)


    #axes2label = [axes[k] for k in ['A1', 'B1', 'C']]
    #label_pos  = [[0.8, 0.9] * 3] 
    #sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure("fig3.pdf")
