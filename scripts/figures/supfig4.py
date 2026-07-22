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

import ccmodels.utils.distances as au

from scipy.stats import ttest_ind_from_stats


def plot_current_indegree(ax_dif, matched_units, matched_connections, rates, vij, nbootstrap, datafile):

    tuned_units = fl.filter_neurons(matched_units, tuning='tuned')
    exc2exconns = fl.synapses_by_id(matched_connections, pre_ids=matched_units['id'], post_ids=tuned_units['id'], who='both')

    in_degrees =  exc2exconns['post_id'].value_counts()

    units_w_indegrees = tuned_units.copy()
    units_w_indegrees['indegree'] = 0
    units_w_indegrees.loc[in_degrees.index.values, 'indegree'] = in_degrees.values

    percentiles = [0, 20, 40, 60, 80, 100] 
    kbounds = np.percentile(in_degrees, percentiles)

    if os.path.exists(datafile):
        data = np.load(datafile) 

        shuf_avgdif  = data["shuf_avgdif"]
        shuf_errdif  = data["shuf_errdif"]
        avgdif       = data["avgdif"]
        errordif     = data["errordif"]
    else:


        shuf_avgdif  = np.zeros(len(percentiles)-1)
        shuf_errdif  = np.zeros(len(percentiles)-1)
        avgdif       = np.zeros(len(percentiles)-1)
        errordif     = np.zeros(len(percentiles)-1)

        id_ns        = []


        for i in range(len(kbounds)-1):
            vij_shuffle = vij[:, np.random.permutation(vij.shape[1])]
            for b in range(nbootstrap):
                unitsboots = units_w_indegrees.sample(frac=1.0, replace=True)
                mask = (kbounds[i] <= unitsboots['indegree'].values) & (unitsboots['indegree'].values <= kbounds[i+1])

                pref_oris = unitsboots.loc[mask, 'pref_ori'].values
                selected_ids = unitsboots.loc[mask, :].index.values

                indiv_currents = curr.get_currents_subset(matched_units, vij, rates, post_ids=selected_ids)
                pred_pref_ori = np.argmax(indiv_currents, axis=1)

                indiv_currents = curr.get_currents_subset(matched_units, vij_shuffle, rates, post_ids=selected_ids)
                shuff_preds   = np.argmax(indiv_currents, axis=1)


                dif = au.unsigned_dist(pred_pref_ori, pref_oris).mean() * np.pi / 8
                avgdif[i]   += dif
                errordif[i] += dif**2 

                dif = au.unsigned_dist(shuff_preds, pref_oris).mean() * np.pi / 8
                shuf_avgdif[i]   += dif
                shuf_errdif[i] += dif**2 

            avgdif[i] /= nbootstrap
            errordif[i] /= nbootstrap 

            shuf_avgdif[i] /= nbootstrap
            shuf_errdif[i] /= nbootstrap 


            _, p = ttest_ind_from_stats(shuf_avgdif[i], shuf_errdif[i], nbootstrap, avgdif[i], errordif[i], nbootstrap, alternative='greater') 
            print(p)

            errordif[i] = np.sqrt(errordif[i] / nbootstrap)
            shuf_errdif[i] = np.sqrt(shuf_errdif[i] / nbootstrap)


        np.savez_compressed(datafile, avgdif=avgdif, errordif=errordif, shuf_avgdif=shuf_avgdif, shuf_errdif=shuf_errdif)
        

    #ax.text(0.5, 1.0, "Data", weight="bold", horizontalalignment='center',transform=ax.transAxes,fontsize=12)
    ax_dif.errorbar(kbounds[1:], avgdif, yerr=errordif, color='gray', marker='o', ms=cr.ms, label='Data')
    ax_dif.errorbar(kbounds[1:], shuf_avgdif, yerr=shuf_errdif, color='black', marker='o', ms=cr.ms, label='Reshuffled control')


    ax_dif.set_xlabel("Observed in-degree")
    ax_dif.set_ylabel(r"$|\hat \theta - \hat \theta_\text{pred}|$")
    #ax_dif.set_yticks([0, np.pi/6, np.pi/5, np.pi/3], [0, 'π/6', "π/5", "π/3"])
    ax_dif.set_yticks([np.pi/6, np.pi/5, np.pi/4], ['π/6', "π/5", "π/4"])
    ax_dif.set_ylim(np.pi/6, np.pi/4+0.1)
    ax_dif.legend(loc='best')




def tuning_prediction_performance(ax_dif, matched_neurons, matched_connections, rates, nbootstrap, datafile): 

    indegrees = np.linspace(25, 1300, 10, endpoint=True, dtype=int)

    if os.path.exists(datafile):
        data = np.load(datafile) 

        avgdif       = data["avgdif"]
        errordif     = data["errordif"]
    else:
        conn_prob = pd.read_csv("data/model/prob_connectomics_cleanaxons.csv", index_col=0)
        ratio =  conn_prob.loc['E', 'X'] / conn_prob.loc['E', 'E']


        tuned_outputs = fl.filter_connections_prepost(matched_neurons, matched_connections,  tuning=['tuned', "tuned"], proofread=['minimum', None])
        avgdif      = np.zeros(len(indegrees))
        errordif     = np.zeros(len(indegrees))

        distangles = np.pi/8 * np.abs(np.arange(-3, 5))
        for i,k in enumerate(indegrees):

            indeg = {}
            indeg["L23"] = k
            indeg["L4"] = int(k*ratio)
            indeg["Total"] = indeg["L23"] + indeg["L4"]

            prob_pref_ori, _, _= curr.sample_prefori(matched_neurons, tuned_outputs, nbootstrap, rates, nsamples=indeg)

            avgdif[i]   = np.sum(distangles * prob_pref_ori['Total'])
            errordif[i] = np.sqrt(np.sum(prob_pref_ori['Total'] * (distangles - avgdif[i])**2) / nbootstrap)

        np.savez_compressed(datafile, avgdif=avgdif, errordif=errordif)

    ax_dif.errorbar(indegrees, avgdif, yerr=errordif, color='black', marker='o', ms=cr.ms)
    ax_dif.set_xlabel("In-degree")
    ax_dif.set_ylabel(r"$|\hat \theta_\text{target} - \hat \theta_\text{emerg}|$")
    ax_dif.set_yticks([0, np.pi/16, np.pi/8, np.pi/4], [0, 'π/16', "π/8", "π/4"])
    #ax_dif.set_ylim(0, 1)

    


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
    #fig = plt.figure(figsize=sty.two_col_size(ratio=3.5), layout='constrained')
    #axes = fig.subplot_mosaic(
    #    """
    #    XY
    #    AB
    #    """, height_ratios=[0.1, 1.]
    #)


    #titles = {'X' : 'Data', 'Y' : 'Virtual postsyn. unit'}
    #for label in 'XY':
    #    ax = axes[label]
    #    ax.set_axis_off()
    #    ax.text(0.5, 1.0, titles[label], weight="bold", horizontalalignment='center', transform=ax.transAxes, fontsize=12)

    #indegrees = np.concatenate((np.arange(50, 400, 50), np.arange(400, 1000, 100)))
    #nbootstrap = 1000
    #plot_current_indegree(axes['A'], matched_neurons, matched_connections, rates, vij, nbootstrap) 
    #tuning_prediction_performance(axes['B'], matched_neurons, matched_connections, rates, nbootstrap) 

    #axes2label = [axes[k] for k in ['A', 'B']]
    #label_pos  = [[0.1, 0.9]] * 2
    #sty.label_axes(axes2label, label_pos)
    #fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


    # Figure 1
    fig1 = plt.figure(figsize=sty.one_col_size(ratio=1.2), layout='constrained')
    axes1 = fig1.subplot_mosaic(
        """
        X
        A
        """, height_ratios=[0.1, 1.])

    axes1['X'].set_axis_off()
    axes1['X'].text(0.5, 1.0, 'Data',
                    weight='bold', ha='center',
                    transform=axes1['X'].transAxes, fontsize=12)

    indegrees = np.concatenate((np.arange(50, 400, 50), np.arange(400, 1000, 100)))
    nbootstrap = 1000

    plot_current_indegree(axes1['A'], matched_neurons, matched_connections, rates, vij, nbootstrap, f"{args.save_destination}/supfig_data_1.npz")

    fig1.savefig(f"{args.save_destination}/{figname}_data.pdf", bbox_inches="tight")





    # Figure 2
    fig2 = plt.figure(figsize=sty.one_col_size(ratio=1.2), layout='constrained')
    axes2 = fig2.subplot_mosaic(
        """
        Y
        B
        """, height_ratios=[0.1, 1.])

    axes2['Y'].set_axis_off()
    axes2['Y'].text(0.5, 1.0, 'Virtual postsyn. unit',
                    weight='bold', ha='center',
                    transform=axes2['Y'].transAxes, fontsize=12)

    tuning_prediction_performance(axes2['B'], matched_neurons, matched_connections, rates, nbootstrap, f"{args.save_destination}/supfig_data_2.npz")

    fig2.savefig(f"{args.save_destination}/{figname}_virtual_postsyn_unit.pdf", bbox_inches="tight")

plot_figure("supfig_kin")
