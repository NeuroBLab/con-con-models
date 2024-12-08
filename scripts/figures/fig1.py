import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import pandas as pd
from scipy.stats import ttest_ind 

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.modelanalysis.utils as mutl
import ccmodels.dataanalysis.currents as dcr
import ccmodels.dataanalysis.currents as curr
import ccmodels.utils.angleutils as au
import ccmodels.dataanalysis.utils as utl


import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()


def show_image(ax, path2im):
    im = Image.open("images/" + path2im)

    ax.imshow(im)
    ax.set_axis_off()


def example_tuning_curves(ax, angles, v1_neurons, rates):

    indices = {'L4':[0,1], 'L23':[0,1]}

    for layer in ['L23', 'L4']:
        neurons_ids = fl.filter_neurons(v1_neurons, layer=layer)
        neurons_ids = neurons_ids['id'].values[indices[layer]]

        for id in neurons_ids:
            rangle = plotutils.shift(rates[id, :])
            ax.plot(angles, rangle,  lw=1, color=cr.lcolor[layer])

    ax.set_xlabel("Δθ")
    ax.set_ylabel("Current")
    plotutils.get_xticks(ax, half=True)


def single_neuron_current(ax, v1_neurons, v1_connections, rates, vij, neuron_id, n_examples=3):
    current = curr.get_input_to_neuron(v1_neurons, v1_connections, neuron_id, vij, rates, shifted=False)
    maxcur = current.max()
    current = current  / maxcur
    rate_neuron = rates[neuron_id,:] / rates[neuron_id,:].max()

    c_rate = cr.pal_extended[3] 
    c_current = cr.pal_extended[1]


    ax.plot(np.arange(8), current,      lw=2, color=c_current, label='Rate')
    ax.plot(np.arange(8), rate_neuron,  lw=2, color=c_rate, label='Current')

    max_values = np.array([np.argmax(current), np.argmax(rate_neuron)]) 
    y_max_values = 0.1 + np.array([current[max_values[0]], rate_neuron[max_values[1]]])
    max_values = max_values

    ax.scatter(max_values, y_max_values, c=[c_current, c_rate], marker='v', s=120) 
    ax.axvline(max_values[0], ls=":", color=c_current)
    ax.axvline(max_values[1], ls=":", color=c_rate)
    ax.set_ylim(0, 1.1)


def plot_matchingprefori_data(ax, matched_neurons, matched_connections, vij, rates, nbootstrap=100):

    bins = np.arange(-3.5, 5.5, 1)
    angles = np.arange(9)

    rates = utl.get_untuned_rate(matched_neurons, rates)

    tuned_neurons = fl.filter_neurons(matched_neurons, tuning='tuned', layer='L23')
    proofread     = fl.filter_neurons(matched_neurons, proofread='ax_clean') 
    tuned_connections = fl.synapses_by_id(matched_connections, pre_ids=proofread['id'], post_ids=tuned_neurons['id'], who='both')

    pref_ori = dcr.fraction_prefori_predicted(matched_neurons, tuned_connections, vij, rates)
    pref_ori = pref_ori[tuned_neurons['id']]

    hist, _ = np.histogram(pref_ori, bins=bins, density=True)
    hist = hist / hist.sum()
    print(hist)
    hist = plotutils.add_symmetric_angle(hist)
    ax.plot(angles, hist, color=cr.pal_extended[1], label='Data')





    av_hist = np.zeros(bins.size-1)
    std_hist = np.zeros(bins.size-1)

    idx_shuffle = np.arange(len(matched_neurons))
    for i in range(nbootstrap): 
        np.random.shuffle(idx_shuffle)
        vij_re = vij[:, idx_shuffle]

        pref_ori_reshuffle = dcr.fraction_prefori_predicted(matched_neurons, tuned_connections, vij_re, rates)
        pref_ori_reshuffle = pref_ori_reshuffle[tuned_neurons['id']]
        hist, _ = np.histogram(pref_ori_reshuffle, bins=bins, density=True)
        hist = hist / hist.sum()
        av_hist += hist 
        std_hist += hist**2 


    print(hist)
    av_hist /= nbootstrap
    std_hist /= nbootstrap

    std_hist = np.sqrt(std_hist - av_hist**2)




    av_hist = plotutils.add_symmetric_angle(av_hist)
    std_hist = plotutils.add_symmetric_angle(std_hist)
    ax.fill_between(angles, av_hist-std_hist, av_hist+std_hist, alpha=0.2, color='gray')
    ax.plot(angles, av_hist, color='gray', label='Reshuffle')


    ax.set_xticks([0, 4, 8], ['-π/2', '0', 'π/2'])
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_xlabel(r"$\hat \theta - \hat \theta(\mu)$")
    ax.set_ylabel("Neuron fraction")
    ax.legend(loc='best', ncols=2)



def in_degree_dist(ax, v1_neurons, v1_connections):
    #filtered_postconns = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[None, 'L23'], cell_type=['exc', 'exc'], proofread=[None, None])
    #in_degrees = filtered_postconns['post_id'].value_counts()

    bins = np.logspace(1, 2.3, 25)
    #nobs = len(in_degrees)
    #weight = np.ones(nobs) / nobs
    #ax.hist(in_degrees, bins=bins, weights=weight, histtype='step',color='#808080',label='No proofr.', density=False)

    filtered_postconns = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[None, 'L23'], cell_type=['exc', 'exc'], proofread=[None, 'dn_clean'])
    in_degrees = filtered_postconns['post_id'].value_counts()
    nobs = len(in_degrees)
    weight = np.ones(nobs) / nobs

    ax.hist(in_degrees, bins=bins, weights=weight, histtype='step',color='#303030',label='Observed', density=False)


    
    l23neurons = fl.filter_neurons(v1_neurons, cell_type='exc')
    dendrites = fl.filter_neurons(l23neurons, proofread='dn_clean')['id'] 
    axons = fl.filter_neurons(l23neurons, proofread='ax_clean')['id']  
    proofconnections = fl.synapses_by_id(v1_connections, pre_ids=axons, post_ids=dendrites, who='both')
    potential_links = len(axons) * len(dendrites)
    p = len(proofconnections) / potential_links

    ax.axvline(x=len(l23neurons) * p ,ls='--',color='r',label='Estimated')


    ax.set_xlabel(r'Number of presyn. neurons')
    ax.set_ylabel(r'Neuron fraction')
    ax.legend(loc=(0.025, 0.65))

    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.set_yticks([0, 0.05, 0.1, 0.15])




def fraction_tuned(ax, data, fstitle=8):
    barw = 0.1
    ybars = [0, barw] 
    offset = 0.05 #To display text

    #Create a Pandas Series which contains the number of tuned neurons in a layer
    #The value is accesed by the key of the pandas dataframe, e.g. n_tuned["L2/3"]
    #tuned_neurons = utl.get_tuned_neurons(data)
    tuned_neurons = fl.filter_neurons(data, tuning="tuned") 
    n_tuned = tuned_neurons.groupby("layer").size()
    total_neurons = data.groupby("layer").size() 

    #TODO check
    #matched_neurons = fl.filter_neurons(data, tuning="matched") 
    #total_neurons = matched_neurons.groupby("layer").size() 

    layers = ["L23", "L4"]

    #Plot those fractions
    for i,layer in enumerate(["L23", "L4"]):
        perc_tuned = n_tuned[layer]/total_neurons[layer]
        ax.barh(ybars[i], perc_tuned, color=cr.lcolor[layer], height=barw)
        ax.text(perc_tuned + offset, ybars[i], f"{round(100*perc_tuned)}%", va="center", ha="left", fontsize=fstitle)
    
    #Configure the axis in a nice way  
    #No spine below, but mark the 100% with a vertical line
    ax.set_yticks(ybars, layers) 
    ax.set_xticks([0, 1], labels=["0", "100%"])
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(1, color="black", lw=3)
    ax.set_xlabel("% of tuned neurons", labelpad=-2., fontsize=fstitle)





def plot_resultant_dist(ax, v1_neurons, rates):
    bins = np.linspace(0, 1, 30)
    bins_centered = 0.5*(bins[1:] + bins[:-1])
    for layer in ['L23', 'L4']:
        tuned_neurons = fl.filter_neurons(v1_neurons, tuning="tuned", layer=layer)
        ratestuned = rates[tuned_neurons['id']]
        cv = mutl.compute_circular_variance(ratestuned, orionly=True)

        #Histogram them, normalizing to count (not by density) 
        hist, _ = np.histogram(cv, bins)
        hist = hist/np.sum(hist)
        ax.step(bins_centered, hist, color = cr.lcolor[layer], label=layer)

    ax.legend(loc='best')
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Circ Var")
    ax.set_ylabel("p(Circ. Var)")


















def plot_figure(figname):
    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.25), layout='constrained')

    #A B B
    #C E G
    #D E G 
    #D F H

    #Top part of figure has two sketches 
    subfigs = fig.subfigures(nrows=2, height_ratios=[1/3,2/3]) 
    axes = {}
    sketches = subfigs[0].subplots(ncols=2, width_ratios=[0.625, 1.])
    axes['A'] = sketches[0]
    axes['B'] = sketches[1]

    #Bottom part: left has the tuning distribution, right has the currents
    subfig_graphs = subfigs[1].subfigures(ncols=2, width_ratios=[0.625, 1])
    subfigs_tuned = subfig_graphs[0].subplots(nrows=2, height_ratios = [0.2, 1.])
    subfigs_currs = subfig_graphs[1].subplots(nrows=2, ncols=2) 

    axes['C'] = subfigs_tuned[0]
    axes['D'] = subfigs_tuned[1]
    axes['E'] = subfigs_currs[0,0]
    axes['F'] = subfigs_currs[1,0]
    axes['G'] = subfigs_currs[0,1]
    axes['H'] = subfigs_currs[1,1]


    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)
    #angles = plotutils.get_angles(kind="centered", half=True)

    show_image(axes['A'], "sketch1.png")
    show_image(axes['B'], "new_neurons.png")

    fraction_tuned(axes['C'], matched_neurons) 
    plot_resultant_dist(axes['D'], matched_neurons, rates)

    single_neuron_current(axes['E'], matched_neurons, matched_connections, rates, vij, 7)
    single_neuron_current(axes['F'], matched_neurons, matched_connections, rates, vij, 2)

    axes['E'].legend(loc=(0.2, 0.7))
    for key in 'EF':
        axes[key].set_xlabel("θ")
        axes[key].set_xticks([0, 4, 8], ['0', 'π/2', 'π'])

    plot_matchingprefori_data(axes['G'], matched_neurons, matched_connections, vij, rates)
    in_degree_dist(axes['H'], units, connections) 


    #Separation between axes
    fig.get_layout_engine().set(wspace=1/72, w_pad=0)

    axes2label = [axes[k] for k in 'ABCEFGH']
    label_pos  = [1.0, 0.9]*2 + [0.8, 0.9] + 3*[0.9, 0.9] + [0.8,0.9] 
    sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure('fig1.pdf')
