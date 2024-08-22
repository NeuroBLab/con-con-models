import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch as Box
import argparse
import pandas as pd

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.currents as curr
import ccmodels.dataanalysis.statistics_extraction as ste


import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_pref_ori(ax, v1_neurons):
    bins = np.arange(-2.5, 9.0, 1)

    for layer in ['L23', 'L4']:
        neurons_layer = fl.filter_neurons(v1_neurons, layer=layer)
        hist, _ = np.histogram(neurons_layer['pref_ori'].values, bins=bins)
        ax.step(bins[1:], hist, color = cr.lcolor[layer], label=layer)

    #ax.legend(loc='upper right', ncols=2)

    ax.set_xlabel('θ')
    ax.set_ylabel('p(θ)')
    ax.set_xticks([0,8], ['0', 'π'])
    ax.set_xlim(-1, 8)

def conn_prob_osi(ax, v1_neurons, v1_connections, half=True):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half)


    #Plot it!
    angles = plotutils.get_angles(kind="centered", half=half)

    for layer in ["L23", "L4"]:
        p = conprob[layer]
        c = cr.lcolor[layer]

        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]

        low_band  = p['mean'] - p['std']
        high_band = p['mean'] + p['std']
        meandata = p['mean']

        low_band  = plotutils.add_symmetric_angle(low_band.values)
        high_band = plotutils.add_symmetric_angle(high_band.values)
        meandata  = plotutils.add_symmetric_angle(meandata.values)

        ax.fill_between(angles, low_band, high_band, color = c, alpha = 0.2)

        ax.plot(angles, meandata, color = c, label = layer)
        ax.scatter(angles, meandata, color = 'black', s=5, zorder = 3)

    ax.axvline(0, color="gray", ls=":")

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel('∆θ')
    ax.set_ylabel("p/p(0)")

    plotutils.get_xticks(ax, max=np.pi, half=True)


def plot_ratedist(ax,layer, v1_neurons, rates):
    bins = np.logspace(-2, 2, 50)

    tuned_neurons = fl.filter_neurons(v1_neurons, layer=layer, tuning='tuned')
    untuned_neurons = fl.filter_neurons(v1_neurons, layer=layer, tuning='untuned')

    tuned_rates = rates[tuned_neurons['id'], :].ravel()
    untuned_rates = rates[untuned_neurons['id'], :].ravel()

    cotuned_rates = np.empty(0)
    orthogo_rates = np.empty(0)

    nangles = rates.shape[1]

    for i in range(nangles//2):
        ortho = (i+nangles//2)%nangles
        id_cotuned = tuned_neurons.loc[tuned_neurons['pref_ori']==i, 'id']
        id_orthogonal =  tuned_neurons.loc[tuned_neurons['pref_ori']==ortho, 'id']

        cotuned_rates = np.concatenate((cotuned_rates, rates[id_cotuned, i]) )
        #orthogo_rates = np.concatenate((orthogo_rates, rates[id_cotuned, ortho]) )
        orthogo_rates = np.concatenate((orthogo_rates, rates[id_orthogonal, i]) )


    labels = ['All', 'Tuned', 'Untuned', 'Cotuned', 'Orthogonal']
    for i,r in enumerate([rates.ravel(), tuned_rates, untuned_rates, cotuned_rates, orthogo_rates]):
        h, edges = np.histogram(r, density=True,  bins=bins)
        dr = edges[1:] - edges[:-1]
        ax.step(edges[:-1], h*dr, label=labels[i])


    ax.set_xscale('log')

    

def plot_synvoldist(ax, layer, v1_neurons, v1_connections):
    bins = np.logspace(-3, 2, 40)
    #bins = np.linspace(-3, 2, 40)


    tuned_links = fl.filter_connections(v1_neurons, v1_connections, layer=layer, tuning='tuned', who='pre')
    untuned_links = fl.filter_connections(v1_neurons, v1_connections, layer=layer, tuning='untuned', who='pre')

    tuned_synvol = tuned_links['syn_volume'].values
    untuned_synvol =  untuned_links['syn_volume'].values

    cotuned_synvol = tuned_links.loc[tuned_links['delta_ori']==0, 'syn_volume'].values
    orthogo_synvol = tuned_links.loc[tuned_links['delta_ori']==4, 'syn_volume'].values

    labels = ['All', 'Tuned', 'Untuned', 'Cotuned', 'Orthogonal']
    for i,r in enumerate([v1_connections['syn_volume'].values, tuned_synvol, untuned_synvol, cotuned_synvol, orthogo_synvol]):
        #h, edges = np.histogram(r, density=True, bins=bins)
        h, edges = np.histogram(r, density=True, bins=bins)
        dv = edges[1:] - edges[:-1]
        ax.step(edges[:-1], h*dv, label=labels[i])



    ax.set_xscale('log')

def plot_sampling_current(ax_mean, ax_std, v1_neurons, v1_connections, rates):
    angles = plotutils.get_angles(kind="centered", half=True)
    nexperiments = 100
    frac = 700 / len(v1_neurons)

    mean_cur, std_cur = curr.bootstrap_system_currents(v1_neurons, v1_connections, rates, nexperiments, frac=frac, replace=False)

    for layer in ['Total', 'L23', 'L4']:
        meancur = mean_cur[layer].mean(axis=1)
        stdcur  = std_cur[layer].mean(axis=1)

        meancur = plotutils.shift(meancur)
        stdcur = plotutils.shift(stdcur)

        ax_mean.plot(angles, meancur, label=layer, color=cr.lcolor[layer])
        ax_std.plot(angles,  stdcur, label=layer, color=cr.lcolor[layer])

        plotutils.get_xticks(ax_mean, max=np.pi, half=True)
        plotutils.get_xticks(ax_std, max=np.pi, half=True)

    
def plot_sampling_current_peaks(ax, v1_neurons, v1_connections, rates):
    frac = 700 / len(v1_neurons)

    current = curr.bootstrap_system_currents_peaks(v1_neurons, v1_connections, rates, frac=frac)
    bins = np.arange(-7.5, 8.5, 1)

    for layer in ['L23', 'L4']:
        pref_ori = np.argmax(current[layer], axis=1)
        pref_ori[pref_ori > 3] = pref_ori[pref_ori > 3] - 8 

        hist, _ = np.histogram(pref_ori, bins=bins)
        print(bins)
        print(hist)
        ax.step(bins[1:], hist, color = cr.lcolor[layer], label=layer)


def tuning_prediction_performance(ax, matched_neurons, matched_connections, rates, nexperiments=1000): 

    angles = plotutils.get_angles(kind="centered", half=True)
    tuned_outputs = fl.filter_connections(matched_neurons, matched_connections, tuning="matched", who="pre") 

    prob_pref_ori  = curr.sample_prefori(matched_neurons, tuned_outputs, nexperiments, rates, nsamples=700)

    #Plot
    for layer in ['Total', 'L23', 'L4']:
        ax.plot(angles, plotutils.shift(prob_pref_ori[layer]), color=cr.lcolor[layer], label=layer)

    plotutils.get_xticks(ax, max=np.pi, half=True)

    ax.set_xlabel("Sampled θ")
    ax.set_ylabel("P(θ)")

    ax.set_yticks([0, 0.5, 1.0])




def plot_figure3(figname):
    # load files
    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    #vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)



    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(height=9.5), layout="constrained")
    ghostax = fig.add_axes([0,0,1,1])
    ghostax.axis('off')

    axes = fig.subplot_mosaic(
        """
        ABXY
        CDZW
        ELLL
        """
    )

    plot_pref_ori(axes['A'], matched_neurons)
    conn_prob_osi(axes['B'], matched_neurons, matched_connections)



    plot_synvoldist(axes['X'], 'L23', matched_neurons, matched_connections)
    plot_synvoldist(axes['Y'], 'L4', matched_neurons, matched_connections)
    plot_ratedist(axes['Z'], 'L23', matched_neurons, rates)
    plot_ratedist(axes['W'], 'L4', matched_neurons, rates)

    axes['X'].set_ylabel('p(V)dV')
    axes['Z'].set_ylabel('p(r)dr')

    for k in 'XY':
        axes[k].set_xlabel('V')
    for k in 'ZW':
        axes[k].set_xlabel('r')


    light = cr.ligthen(cr.lcolor['L23'], 1, 0.8)[0]
    boxL23 = Box(xy=[0.55,  0.34], width=0.16, height=0.63, boxstyle='round, pad=0.04', lw=0, fc=light, transform=fig.transFigure)
    light = cr.ligthen(cr.lcolor['L4'], 1, 0.8)[0]
    boxL4= Box(xy=[0.81,  0.34], width=0.15, height=0.63, boxstyle='round, pad=0.04', lw=0, fc=light, transform=fig.transFigure)
    ghostax.add_patch(boxL23)
    ghostax.add_patch(boxL4)



    plot_sampling_current(axes['C'], axes['D'], matched_neurons, matched_connections, rates)
    #plot_sampling_current_peaks(leftaxes[2,0], matched_neurons, matched_connections, rates)

    tuning_prediction_performance(axes['E'], matched_neurons, matched_connections, rates)

    axes['L'].axis('off')

    handles, labels = axes['X'].get_legend_handles_labels()
    legend_dist = axes['L'].legend(handles, labels, ncols=3, loc=(0.3,0), alignment='left')

    handles, labels = axes['C'].get_legend_handles_labels()
    axes['L'].legend(handles, labels, ncols=1, loc=(0,0.0), alignment='left')
    axes['L'].add_artist(legend_dist)

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure3("fig2paper.pdf")