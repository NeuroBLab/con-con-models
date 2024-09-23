import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import pandas as pd

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.modelanalysis.utils as mutl
import ccmodels.dataanalysis.currents as dcr
import ccmodels.dataanalysis.currents as curr
import ccmodels.utils.angleutils as au


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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


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


def single_neuron_current(ax, ax_curr, angles, v1_neurons, v1_connections, rates, vij, neuron_id, n_examples=10):
    current = curr.get_input_to_neuron(v1_neurons, v1_connections, neuron_id, vij, rates, shifted=False)
    current = plotutils.shift(current) / current.max()
    rate_neuron = plotutils.shift(rates[neuron_id,:]) / rates[neuron_id,:].max()


    ax.plot(angles, current,      lw=1, color='grey')
    ax.plot(angles, rate_neuron,  lw=1, color='green')

    max_values = np.array([np.argmax(current), np.argmax(rate_neuron)]) 
    y_max_values = 0.1 + np.array([current[max_values[0]], rate_neuron[max_values[1]]])
    max_values = au.index_to_angle(max_values)

    #TODO two shades of gray would be better
    colors = ['gray', 'green']


    ax.scatter(max_values, y_max_values, c=colors, marker='v') 
    ax.set_ylim(0, 1.1)
 
    pre_ids = fl.connections_to(neuron_id, v1_connections).values
    pre_ids = pre_ids[:n_examples]
    for i in range(n_examples):
        current = curr.get_currents_subset(v1_neurons, vij, rates, post_ids=[neuron_id], pre_ids=[pre_ids[i]], shift=False)
        current = plotutils.add_symmetric_angle(current[0,:])
        ax_curr.plot(angles, current, lw=1, color='grey')


    #ax.axvline(au.index_to_angle(v1_neurons.loc[neuron_id, 'pref_ori']), ls=':', color='gray')




def plot_matchingprefori_data(ax, angles, matched_neurons, matched_connections, vij, rates):
    tuned_neurons = fl.filter_neurons(matched_neurons, tuning='tuned', layer='L23')
    tuned_connections = fl.synapses_by_id(matched_connections, pre_ids=matched_neurons['id'], post_ids=tuned_neurons['id'], who='both')

    pref_ori = dcr.fraction_prefori_predicted(matched_neurons, tuned_connections, vij, rates)
    pref_ori = pref_ori[tuned_neurons['id']]

    idx_shuffle = np.arange(len(matched_neurons))
    np.random.shuffle(idx_shuffle)
    vij_re = vij[:, idx_shuffle]

    pref_ori_reshuffle = dcr.fraction_prefori_predicted(matched_neurons, tuned_connections, vij_re, rates)
    pref_ori_reshuffle = pref_ori_reshuffle[tuned_neurons['id']]

    bins = np.arange(-3.5, 5.5, 1)
    for po,color,lab in zip([pref_ori, pref_ori_reshuffle], ['green', 'gray'], ['Data', 'Reshuffle']):
        hist, _ = np.histogram(po, bins=bins, density=True)
        hist = hist / hist.sum()
        hist = plotutils.add_symmetric_angle(hist)
        ax.plot(angles, hist, color=color, label=lab)


    plotutils.get_xticks(ax, max=np.pi, half=True)
    ax.set_yticks([0, 0.1, 0.2])

    ax.set_xlabel("Sampled θ")
    ax.set_ylabel("P(θ)")

def in_degree_dist(ax, v1_neurons, v1_connections):

    unit_table = v1_neurons.copy()
    connections_table = v1_connections.copy()


    # put pref orientation of not-selective neurons to nan
    mask_not_selective=(unit_table['tuning_type']=='not_selective')
    unit_table.loc[mask_not_selective, 'pref_ori'] = np.nan


    # focus on orientation, take preferred orientation mod pi
    mask_selective=(unit_table['tuning_type']=='orientation')|(unit_table['tuning_type']=='direction')
    unit_table.loc[mask_selective, 'tuning_type'] ='selective'


    # Filter unit_table to consider only rows where axon_proof is not 'non'
    filtered_unit_table = fl.filter_neurons(unit_table, proofread='minimum')

    # Join connections_table with filtered_unit_table on pre_pt_root_id and pt_root_id
    merged_table = pd.merge(connections_table, filtered_unit_table, left_on='pre_id', right_on='id', how='inner')

    # Group by post_pt_root_id and count unique pre_pt_root_id values for each group
    pre_pt_count_per_post = merged_table.groupby('post_id')['pre_id'].nunique().reset_index()
    pre_pt_count_per_post.columns = ['post_id', 'pre_count']


    ax.hist(pre_pt_count_per_post['pre_count'].values,np.arange(0,10**3,1),density=True,histtype='step',color='k',label='Observed')
    ax.axvline(x=700,ls='--',color='r',label='Estimated')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Number of presyn. neurons')
    ax.set_ylabel(r'Fraction')
    ax.legend(loc=(0.4, 0.6))

    ax.set_yticks([1e-4, 1e-2, 1e-0])
    ax.set_ylim(1e-4, 1.0)




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
        ax.step(bins_centered, hist, color = cr.lcolor[layer])


















def plot_figure(figname):
    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.75), layout='constrained')

    axes = fig.subplot_mosaic(
        """
        ABC
        ABD
        EFG
        HIJ
        """, gridspec_kw={"width_ratios":[1, 1, 1.3], "height_ratios":[0.3, 1.0, 0.8, 0.8]}
    )

    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)
    angles = plotutils.get_angles(kind="centered", half=True)

    show_image(axes['A'], "network_schema.png")
    show_image(axes['B'], "3d_reconstruction.png")

    single_neuron_current(axes['E'], axes['H'], angles, matched_neurons, matched_connections, rates, vij, 0)
    single_neuron_current(axes['F'], axes['I'], angles, matched_neurons, matched_connections, rates, vij, 2)

    axes['H'].set_xlabel("Δθ")
    plotutils.get_xticks(axes['H'], half=True)
    axes['E'].set_xticks([])

    fraction_tuned(axes['C'], matched_neurons) 
    plot_resultant_dist(axes['D'], matched_neurons, rates)

    plot_matchingprefori_data(axes['G'], angles, matched_neurons, matched_connections, vij, rates)
    in_degree_dist(axes['J'], units, connections) 


    #Separation between axes
    fig.get_layout_engine().set(wspace=1/72, w_pad=0)

    axes2label = [axes[k] for k in 'ABDEFGJ']
    label_pos  = [0.9, 0.9]*7 
    sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


#plot_figure('fig1.pdf')




"""
def plot_figure():
    sty.master_format()
    fig = plt.figure(figsize=sty.slide_size(0.25, 0.5), layout='constrained')
    ax = plt.gca()


    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)
    angles = plotutils.get_angles(kind="centered", half=True)

    plot_matchingprefori_data(ax, angles, matched_neurons, matched_connections, vij, rates)

    ax.legend()

    fig.savefig(f"predict_tuning.pdf",  bbox_inches="tight")

plot_figure()
"""