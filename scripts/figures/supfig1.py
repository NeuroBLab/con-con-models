import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import pandas as pd
from itertools import product

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.statistics_extraction as ste


import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils


def prob_angles_proofread(axes, v1_neurons, v1_connections):

    #Plot it!
    angles = plotutils.get_angles(kind="centered", half=True)

    #cases = [['non', 'non'], ['clean', 'clean'], ['extended', 'extended'], ['checked', 'none']] 
    cases = [['non', 'non'], ['clean', 'clean'], ['checked', 'none']] 


    colors = plt.cm.tab10(np.linspace(0, 1, 4))
             
    for i,case in enumerate(cases):
        #Get the data to be plotted 
        print(case, colors[i]) 
        c = colors[i] if i < 2 else colors[i+1]
        prob = ste.bootstrap_prob_tuned2tuned(v1_neurons, v1_connections, pre_layer='L23', half=True, proofread=[f'ax_{case[0]}', f'dn_{case[1]}']) 

        #Normalize by p(delta=0), which is at index 3
        prob.loc[:, ["mean", "std"]] = prob.loc[:, ["mean", "std"]] #/p.loc[3, "mean"]

        meandata  = plotutils.add_symmetric_angle(prob['mean'].values)
        errordata = plotutils.add_symmetric_angle(prob['std'].values)

        #"""
        meandata = 0.5 * (meandata + meandata[::-1])
        errordata = 0.5 * (errordata + errordata[::-1])
        nangles = len(meandata)//2

        meandata = meandata[nangles:]
        errordata = errordata[nangles:]
        x = angles[nangles:]
        #"""
        #x = angles


        print(meandata)
        axes[0].errorbar(x, meandata, yerr=errordata, color=c)
        axes[0].axvline(0, color="gray", ls=":")

        norm = np.sum(meandata)
        axes[1].errorbar(x, meandata/norm, yerr=errordata/norm, color=c)
        axes[1].axvline(0, color="gray", ls=":")


        #Then just adjust axes and put a legend
        axes[1].tick_params(axis='both', which='major')
        axes[1].set_xlabel('∆θ')

        axes[0].set_xticks([0, np.pi/2], ['0', 'π/2'])
        axes[0].set_xticklabels([])
        axes[1].set_xticks([0, np.pi/2], ['0', 'π/2'])

        axes[0].set_ylabel("p")
        axes[1].set_ylabel(r"p/$\Sigma _\theta$p($\theta$)")




def prob_connectomics_proofread(axes, v1_neurons, v1_connections):
    cases = [['non', 'non'], ['clean', 'clean'], ['extended', 'extended'], ['checked', 'none']] 
    bars = ['EE', 'EI', 'EX', 'IE', 'II', 'IX']
    labels = [fr"{lab}$\leftarrow${lab}" for lab in ['non', 'clean', 'extnd']] + [r"all $\leftarrow$ clean/extnd"]
    x_labels = [fr"{x[0]}$\leftarrow${x[1]}" for x in ['EE', 'EI', 'EX', 'IE', 'II', 'IX']] 

    bar_width = 0.15  # Width of the bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(cases)))


    prob = np.empty((len(cases), len(bars)))     
    error = np.empty((len(cases), len(bars)))     

    for i,case in enumerate(cases):
        print(case, colors[i], labels[i])
        ptable_mean, ptable_std = ste.estimate_conn_prob_connectomics_2(v1_neurons, v1_connections, proof=[f'ax_{case[0]}', f'dn_{case[1]}'])

        for j,b in enumerate(bars):
            prob[i,j] = ptable_mean.loc[b[0], b[1]]
            error[i,j] = ptable_std.loc[b[0], b[1]]

            if prob[i,j] > 0:
                index = j + i * bar_width - (len(cases) * bar_width) / 2

                axes[0].bar(index, prob[i,j], bar_width, yerr=error[i,j], alpha=0.7, color=colors[i], label=labels[i])
                axes[1].bar(index, prob[i,j]/prob[i,0], bar_width, yerr=error[i,j]/prob[i,0], alpha=0.7, color=colors[i])

    axes[0].set_yscale('log')
    axes[0].set_yticks([1e-4, 1e-2, 1])
    axes[0].set_ylim([0.0001, 0.5])

    axes[1].set_yscale('log')
    axes[1].set_yticks([1e-4, 1e-2, 1])
    axes[1].set_ylim([0.01, 50])

    axes[1].set_xticks(np.arange(len(x_labels)))
    axes[1].set_xticklabels(x_labels, rotation=0, ha='center')
    axes[0].set_xticklabels([], rotation=0, ha='center')

    axes[0].set_ylabel("p")
    axes[1].set_ylabel(r"p/p(E$\leftarrow$E)")

    return axes[0].get_legend_handles_labels()



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



    sty.master_format()
    #fig, axes = plt.subplots(nrows=3, ncols=2, figsize=sty.two_col_size(ratio=2), layout="constrained", height_ratios=[0.2,1,1])
    fig, axes = plt.subplot_mosaic(
        """
        AA
        BD
        CE
        """,
        figsize=sty.two_col_size(ratio=2), layout="constrained", height_ratios=[0.2,1,1], gridspec_kw = {'wspace' : 0.05})

    #for i in range(2):
    #    axes[0,i].set_axis_off()
    axes['A'].set_axis_off()

    handles, labels = prob_connectomics_proofread([axes['B'], axes['C']], units, connections)
    prob_angles_proofread([axes['D'], axes['E']], matched_neurons, matched_connections)

    #Remove duplicates in labels/handels by converting to dict and create legend
    by_label = dict(zip(labels, handles))
    axes['A'].legend(by_label.values(), by_label.keys(), ncols=4, loc='center left')


    sty.label_axes([axes[c] for c in 'BDCE'], textpos=4*[0.9, 0.9])

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")

plot_figure("supfig1.pdf")