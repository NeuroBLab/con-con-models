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

def plot_dist_inputs(ax1, ax2, v1_neurons, v1_connections, rates):

    bins = np.arange(-4.5, 5.5, 1)
    h, _ = np.histogram(v1_connections['delta_ori'],  bins=bins, density=False)
    h = h / h.sum()
    h[0] = h[-1] #Between -4.5 and -3.5 value is always 0. To make it periodic, match with the last one

    centered_bins = 0.5 * (bins[:-1] + bins[1:])

    ax1.plot(centered_bins, h, color='k')
    ax1.scatter(centered_bins, h, color='k', s=cr.ms, zorder=3)
    ax1.set_xticks([-4, 0, 4], ["-π/2", "0", "π/2"])    

    ax1.set_xlabel(r'$\hat \theta_\text{post} - \theta$')
    ax1.set_ylabel("Fract. neurons")

    ax1.set_ylim(0, 0.17)

    preids = v1_connections['pre_id'].values
    pref_ori = v1_neurons.loc[preids, 'pref_ori']

    #Bar widths
    labels = ["Pref. ori.", "Orthogonal"]
    deltas = [0, 4]
    ci =     [0,2]

    bins = np.logspace(-2, 1.5, 60)

    for i in range(2):
        conns = v1_connections.loc[(v1_connections['delta_ori'] == deltas[i]) | (v1_connections['delta_ori'] == -deltas[i])]
        h, _ = np.histogram(conns['syn_volume'], bins=bins)
        h = h / h.sum()

        centered_bins = 0.5 * (bins[:-1] + bins[1:]) 
        ax2.step(centered_bins, h, color = cr.pal_qualitative[ci[i]], label=labels[i])

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.set_ylim(1e-4, 1e-1)

    ax2.set_xlabel('Volume (mean normalized)')
    ax2.set_ylabel("Frac. Synapses")
    ax2.legend(loc=(0.2, 0.9), ncol=2, fontsize=8)

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=100):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections)
    meandata  = {}
    errordata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[0, "mean"]
        meandata[layer]  = p['mean'].values 
        errordata[layer]  = p['std'].values 

    return meandata, errordata

def conn_prob_osi(ax, probmean, proberr):

    #Plot it!
    angles = np.linspace(0, np.pi/2, 5)

    for layer in ['L23', 'L4']:
        low_band  = probmean[layer] - proberr[layer]
        high_band = probmean[layer] + proberr[layer]
        c = cr.lcolor[layer]

        ax.fill_between(angles, low_band, high_band, color = c, alpha = 0.2)
        ax.plot(angles, probmean[layer], color = c, label = layer)
        ax.scatter(angles, probmean[layer], color = cr.dotcolor[layer], s=cr.ms, zorder = 3)

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_ylim(0.5, 1.1)
    ax.set_xlabel(r"$|\theta -  \hat \theta_\text{post}|$")
    ax.set_ylabel("Conn. Prob\n(Normalized)")

    ax.legend(loc='best')

    ax.set_xticks([0, np.pi/4, np.pi/2], ["0", "π/4", "π/2"])


def plot_sampling_current(ax, ax_normalized, v1_neurons, v1_connections, rates, indegree, nexperiments=1000):
    angles = plotutils.get_angles(kind="centered", half=True)

    #Compute the currents in the system
    #mean_cur = curr.bootstrap_system_currents_shuffle(v1_neurons, v1_connections, rates, nexperiments, frac=frac)
    mean_cur, std_cur = curr.bootstrap_mean_current(indegree, v1_neurons, v1_connections, rates, nexperiments)

    #Total current is shown just in the "unnormalized" version. Also we need to obtain
    #the global total current to normalize according to it

    #Then show L23 and L4 currents for unnormalized and normalized versions
    for layer in ['L23', 'L4', 'Total']:
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
    #ax.set_ylabel('μ(Δθ)')
    #ax_normalized.set_ylabel('μ(Δθ)/μ(0)')
    ax.set_ylabel('Syn. Current')
    ax_normalized.set_ylabel('Syn. Current\n(Normalized)')

    
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
    #tuned_outputs = fl.filter_connections(matched_neurons, matched_connections, tuning="matched", who="post") 
    tuned_outputs = fl.filter_connections_prepost(matched_neurons, matched_connections, layer=['L23', 'L23'], tuning=['matched', "matched"])
    prob_pref_ori, currents  = curr.sample_prefori(matched_neurons, tuned_outputs, nexperiments, rates, nsamples=indegree)
    

    #Plot
    for layer in ['Total', 'L23', 'L4']:
        prob    = plotutils.shift(prob_pref_ori[layer])
        proberr = plotutils.shift(prob_pref_ori[layer + "_error"])

        ax.fill_between(angles, prob - proberr, prob + proberr, alpha = 0.5, color=cr.lcolor[layer])
        ax.plot(angles, prob, color=cr.lcolor[layer], label=layer)
        ax.scatter(angles, prob, color=cr.dotcolor[layer], zorder=3, s=cr.ms) 

        #prob = plotutils.shift(currents[layer])
        #ax.plot(angles, prob, color=cr.lcolor[layer], label=layer)


    shuffled_neurons = matched_neurons.copy()
    shuffled_neurons['pref_ori'] = shuffled_neurons['pref_ori'].sample(frac=1).values 
    null_pref_ori, null_currents = curr.sample_prefori(shuffled_neurons, tuned_outputs, nexperiments, rates, nsamples=indegree)
    null_prob    = plotutils.shift(null_pref_ori["Total"])
    null_proberr = plotutils.shift(null_pref_ori["Total_error"])

    ax.fill_between(angles, null_prob - null_proberr, null_prob + null_proberr, alpha = 0.5, color='purple')
    ax.plot(angles, null_prob, c='purple', label='Shuffled TOTAL')

    ax.set_xlabel(r"$\hat \theta_\text{target} - \hat \theta_\text{emerg}$")
    ax.set_ylabel("Fract. neurons")

    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_yticks([0, 0.25, 0.5])

    ax.legend(loc="best")


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
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.2), layout="constrained")

    axes = fig.subplot_mosaic(
        """
        ABC
        XXX
        DEF
        """, height_ratios=([1.0, 1.5, 1])
    )

    #Given an exc -> exc in-degree kee, how many inputs does a neuron receive in total?
    #Compute it using the probabilities obtained from the data 
    kee = 150 
    conn_prob = pd.read_csv("data/model/prob_connectomics_cleanaxons.csv", index_col=0)
    indegree = int(kee * (1.0 + conn_prob.loc['E', 'X'] / conn_prob.loc['E', 'E']))

    show_image(axes["X"], "sketchsampling.png")

    plot_dist_inputs(axes['A'], axes['C'], matched_neurons, matched_connections, rates)
    probmean, proberr = compute_conn_prob(units, connections)
    conn_prob_osi(axes['B'], probmean, proberr)
    plot_sampling_current(axes["D"], axes["E"], matched_neurons, matched_connections, rates, indegree)
    tuning_prediction_performance(axes['F'], matched_neurons, matched_connections, rates, indegree)


    axes2label = [axes[k] for k in ['A', 'B', 'C', 'X', 'D', 'E', 'F']]
    label_pos  = [[0.05, 0.95]] * 7 
    sty.label_axes(axes2label, label_pos)
    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure("fig3.pdf")

#TODO ADD RANDOM CONTROL TO LAST PANEL