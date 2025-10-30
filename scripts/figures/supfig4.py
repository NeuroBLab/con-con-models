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


def plot_current_indegree(ax, matched_units, matched_connections, rates, vij, nbootstrap):

    tuned_units = fl.filter_neurons(matched_units, tuning='tuned')
    exc2exconns = fl.synapses_by_id(matched_connections, pre_ids=matched_units['id'], post_ids=tuned_units['id'], who='both')

    in_degrees =  exc2exconns['post_id'].value_counts()

    units_w_indegrees = tuned_units.copy()
    units_w_indegrees['indegree'] = 0
    units_w_indegrees.loc[in_degrees.index.values, 'indegree'] = in_degrees.values

    percentiles = [0, 20, 40, 60, 80, 100] 
    kbounds = np.percentile(in_degrees, percentiles)

    fraccorrect = np.zeros(len(percentiles)-1)
    fracerror   = np.zeros(len(percentiles)-1)

    for i in range(len(kbounds)-1):
        for b in range(nbootstrap):
            unitsboots = units_w_indegrees.sample(frac=1.0, replace=True)
            mask = (kbounds[i] <= unitsboots['indegree'].values) & (unitsboots['indegree'].values <= kbounds[i+1])


            selected_ids = np.argwhere(mask)[:,0]
            pref_oris = unitsboots.iloc[selected_ids]['pref_ori'].values
            selected_ids = unitsboots.index.values[selected_ids]

            indiv_currents = curr.get_currents_subset(matched_units, vij, rates, post_ids=selected_ids)
            pred_pref_ori = np.argmax(indiv_currents, axis=1)

            fraccorrect[i] += (pred_pref_ori == pref_oris).sum() / len(pref_oris) 
        
        fraccorrect[i] /= nbootstrap
        fracerror[i] = np.sqrt(fraccorrect[i] * (1-fraccorrect[i])/nbootstrap)

    ax.text(0.5, 1.0, "Data", weight="bold", horizontalalignment='center',transform=ax.transAxes,fontsize=12)
    ax.errorbar(kbounds[1:], fraccorrect, yerr=fracerror, color='black', marker='o', ms=cr.ms)
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Fraction predicted")

def tuning_prediction_performance(ax, matched_neurons, matched_connections, rates, nbootstrap): 

    conn_prob = pd.read_csv("data/model/prob_connectomics_cleanaxons.csv", index_col=0)
    ratio =  conn_prob.loc['E', 'X'] / conn_prob.loc['E', 'E']


    tuned_outputs = fl.filter_connections_prepost(matched_neurons, matched_connections,  tuning=['tuned', "tuned"], proofread=['minimum', None])
    indegrees = np.linspace(25, 1300, 10, endpoint=True, dtype=int)
    fraccorrect = np.zeros(len(indegrees))
    fracerror   = np.zeros(len(indegrees))
    for i,k in enumerate(indegrees):

        indeg = {}
        indeg["L23"] = k
        indeg["L4"] = int(k*ratio)
        indeg["Total"] = indeg["L23"] + indeg["L4"]

        prob_pref_ori, _, _= curr.sample_prefori(matched_neurons, tuned_outputs, nbootstrap, rates, nsamples=indeg)
        fraccorrect[i] = prob_pref_ori["Total"][3]
        fracerror[i] = np.sqrt(fraccorrect[i] * (1-fraccorrect[i])/nbootstrap)
    
    ax.text(0.5, 1.0, "Virtual postsyn. unit", weight="bold", horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.errorbar(indegrees, fraccorrect, yerr=fracerror, color='black', marker='o', ms=cr.ms)
    ax.set_xlabel("In-degree")
    ax.set_ylim(0, 1)

    


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
    fig = plt.figure(figsize=sty.two_col_size(ratio=2.5), layout='constrained')
    #ghostax = fig.add_axes([0,0,1,1])
    
    #ghostax.axis('off')

    axes = fig.subplot_mosaic(
        """
        AB
        """
    )


    indegrees = np.concatenate((np.arange(50, 400, 50), np.arange(400, 1000, 100)))
    nbootstrap = 1000
    plot_current_indegree(axes['A'], matched_neurons, matched_connections, rates, vij, nbootstrap) 
    tuning_prediction_performance(axes['B'], matched_neurons, matched_connections, rates, nbootstrap) 

    axes2label = [axes[k] for k in ['A', 'B']]
    label_pos  = [[0.05, 0.95]] * 2 
    sty.label_axes(axes2label, label_pos)
    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure("supfig_kin.pdf")
