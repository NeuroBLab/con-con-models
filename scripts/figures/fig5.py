import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import sys
import os 
sys.path.append(os.getcwd())
import argparse

import ccmodels.modelanalysis.utils as utl

import ccmodels.utils.angleutils as au

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.utils as plotutils
import ccmodels.plotting.color_reference as cr

#def diff_emergent2target_prefori(ax, pref_ori, target_ori, color):
def diff_emergent2target_prefori(ax, diff_ori, color):


    bins = np.arange(-4.5, 5.5)

    hist, edges = np.histogram(diff_ori, bins=bins)

    hist[0] = hist[-1] #Boundary conditions for angle

    hist = hist / len(diff_ori)
    ax.plot(bins[:-1]+0.5, hist, marker='.', color=color)

    #ax.hist(diff_ori, bins=bins,  weights=w, density=False, histtype='step', color=color)

    ax.set_xlabel(r"$\hat \theta _\text{targt}- \hat \theta _\text{emerg}$")
    ax.set_ylabel('Count')
    ax.set_xticks([-4, 0, 4], ['-π/2', '0', 'π/2'])
    ax.set_yticks([0, 0.25, 0.5])

def plot_ratedist(ax, re, color):
    bins = np.linspace(0.01, 10, 50)
    w = np.ones(re.size) / re.size

    ax.hist(re.ravel(), density=False,  weights=w, histtype='step', bins=bins, color=color)

    ax.set_xlabel("Rate (spk/s)")
    ax.set_ylabel('Fract. of neurons')
    #ax.set_xscale('log')
    return


#def circular_variance(ax, re, color):
def circular_variance(ax, cved, color):
    bins = np.linspace(0,1,50)

    #cveo, cved = utl.compute_circular_variance(re, orionly=True)    

    ax.hist(cved, bins=bins, density=True, color=color, histtype='step')

    ax.set_xlabel("CirVar")
    ax.set_ylabel("p(CirVar)")


def compute_conn_prob(v1_neurons, v1_connections, half=True):
    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, frac=0.01)

    #Plot it!
    data = {'L23':None, 'L4':None}

    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]

        data[layer]  = plotutils.add_symmetric_angle(p['mean'].values)


    return data 


def conn_prob_osi(axL23, axL4, meandata, error, colorL23, colorL4, half=True):

    #Plot it!
    angles = plotutils.get_angles(kind="centered", half=half)
    axes = {'L23': axL23, 'L4': axL4}
    colors = {'L23': colorL23, 'L4': colorL4}
    plots = {'L23':None, 'L4':None}

    for layer in ["L23", "L4"]:
        low_band  = meandata[layer] - error[layer]
        high_band = meandata[layer] + error[layer]

        axes[layer].fill_between(angles, low_band, high_band, color = colors[layer], alpha = 0.2)
        plots[layer], = axes[layer].plot(angles, meandata[layer], color = colors[layer])

        axes[layer].axvline(0, color="gray", ls=":")

        #Then just adjust axes and put a legend
        axes[layer].tick_params(axis='both', which='major')
        axes[layer].set_xlabel(r"$\hat \theta _\text{targt}- \theta$")


        plotutils.get_xticks(axes[layer], max=np.pi, half=True)
    
    return plots['L23'], plots['L4']
"""
def conn_prob_osi(axL23, axL4, v1_neurons, v1_connections, colorL23, colorL4, half=True):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half)

    #Plot it!
    angles = plotutils.get_angles(kind="centered", half=half)
    axes = {'L23': axL23, 'L4': axL4}
    colors = {'L23': colorL23, 'L4': colorL4}
    plots = {'L23':None, 'L4':None}

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

        axes[layer].fill_between(angles, low_band, high_band, color = colors[layer], alpha = 0.2)
        plots[layer], = axes[layer].plot(angles, meandata, color = colors[layer])


        axes[layer].axvline(0, color="gray", ls=":")

        #Then just adjust axes and put a legend
        axes[layer].tick_params(axis='both', which='major')
        axes[layer].set_xlabel(r"$\hat \theta _\text{targt}- \theta$")


        plotutils.get_xticks(axes[layer], max=np.pi, half=True)
    
    return plots['L23'], plots['L4']
"""

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 5''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname, generate_data=True):

    nexp = 10

    # load files
    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)


    filename = 'best_ale'
    sty.master_format()
    fig, axes = plt.subplot_mosaic(
    """
    ABC
    DEL
    """,
    figsize=sty.two_col_size(height=9.5), layout='constrained') 

    colors23 = [cr.lcolor['L23']] + cr.darken(cr.lcolor['L23'], 3, 0.25)
    colors4 = [cr.lcolor['L4']] + cr.darken(cr.lcolor['L4'], 3, 0.2)

    plots = []

    for i, reshuffle_mode in enumerate(['', 'alltuned', 'L23tuned', 'L4tuned']):

        c23 = colors23[i]
        c4  = colors4[i]

        if generate_data:

            diff_ori = np.empty(0)
            allrates = np.empty(0)
            allcircv = np.empty(0)
            probmean = {'L23' : np.zeros(9), 'L4' : np.zeros(9)} 
            proberr = {'L23' : np.zeros(9), 'L4' : np.zeros(9)} 

            for j in range(nexp):
                if len(reshuffle_mode) > 1:
                    filepath = f'{filename}_{reshuffle_mode[:3]}_{j}'
                else:
                    filepath = f'{filename}_{j}'
                units_sample, connections_sample, rates_sample, n_neurons, target_ori = utl.load_synthetic_data(filepath)
                QJ = loader.get_adjacency_matrix(units_sample, connections_sample)
                ne, ni, nx = n_neurons

                re = rates_sample[:ne, :]
                ri = rates_sample[ne:ne+ni, :]
                rx = rates_sample[ne+ni:, :]
                
                exc_pref_ori = fl.filter_neurons(units_sample, cell_type='exc', layer='L23')['pref_ori'].values
                diff_ori = np.concatenate((diff_ori, au.signed_angle_dist_vectorized(target_ori, exc_pref_ori)))

                allrates = np.concatenate((allrates, re.ravel()))

                cveo, cved = utl.compute_circular_variance(re, orionly=True)    
                allcircv = np.concatenate((allcircv, cved))

                means = compute_conn_prob(units_sample, connections_sample)
                for layer in ['L23', 'L4']:
                    probmean[layer] += means[layer]
                    proberr[layer] += means[layer]**2

            for layer in ['L23', 'L4']:
                probmean[layer] /= nexp
                proberr[layer] /= nexp
                proberr[layer] -= probmean[layer]**2
                proberr[layer] = np.sqrt(proberr[layer])

            np.save(f"{args.save_destination}/{figname}_{i}_angl_data", diff_ori)
            np.save(f"{args.save_destination}/{figname}_{i}_rate_data", allrates)
            np.save(f"{args.save_destination}/{figname}_{i}_circ_data", allcircv)
            np.save(f"{args.save_destination}/{figname}_{i}_probmeanL23", probmean['L23'])
            np.save(f"{args.save_destination}/{figname}_{i}_proberroL23", proberr['L23'])
            np.save(f"{args.save_destination}/{figname}_{i}_probmeanL4", proberr['L4'])
            np.save(f"{args.save_destination}/{figname}_{i}_proberroL4", proberr['L4'])

        else:
            probmean = {}
            proberr  = {}
            currmean = {}
            currerr  = {}

            diff_ori = np.save(f"{args.save_destination}/{figname}_{i}_angl_data")
            allrates = np.save(f"{args.save_destination}/{figname}_{i}_rate_data", allrates)
            allcircv = np.save(f"{args.save_destination}/{figname}_{i}_circ_data", allrates)
            probmean['L23'] = np.save(f"{args.save_destination}/{figname}_{i}_probmeanL23", probmean['L23'])
            proberr['L23']  = np.save(f"{args.save_destination}/{figname}_{i}_proberroL23", proberr['L23'])
            probmean['L4']  = np.save(f"{args.save_destination}/{figname}_{i}_probmeanL4", proberr['L4'])
            proberr['L4']   = np.save(f"{args.save_destination}/{figname}_{i}_proberroL4", proberr['L4'])



        #diff_emergent2target_prefori(axes['A'], exc_pref_ori, target_ori, c23)    
        diff_emergent2target_prefori(axes['A'], diff_ori, c23)    

        #plot_ratedist(axes['B'], re, c23)
        plot_ratedist(axes['B'], allrates, c23)

        #circular_variance(axes['C'], re, c23)
        circular_variance(axes['C'], allcircv, c23)


        #p1, p2 = conn_prob_osi(axes['D'], axes['E'], units_sample, connections_sample, c23, c4)
        p1, p2 = conn_prob_osi(axes['D'], axes['E'], probmean, proberr, c23, c4)
        plots.append((p1, p2))

    axes['L'].set_axis_off()
    axes['L'].legend(plots, ['Original', 'Reshuffled', 'L23 reshuffled.', 'L4 reshuffled'], loc=(0.0, 0.5), handlelength=3.0, handler_map={tuple: HandlerTuple(ndivide=None)})


    axes2label = [axes[key] for key in 'ABCDE']
    label_pos  = [0.8, 0.9] * 5 
    sty.label_axes(axes2label, label_pos)
    

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")

plot_figure("fig5.pdf")