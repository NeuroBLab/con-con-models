import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.modelanalysis.utils as mutl
import ccmodels.dataanalysis.utils as utl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils



def show_image(ax, path2im):
    im = Image.open("images/" + path2im)

    ax.imshow(im)


def example_tuning_curve(ax, v1_neurons, rates, error_rates, layer='L23'):

    neurons_ids = fl.filter_neurons(v1_neurons, layer=layer, tuning='tuned')
    neurons_ids = neurons_ids['id']

    ids = [8, 10, 11]

    for c,id in enumerate(ids):
        #TODO do not shift here...
        rangle = rates[neurons_ids[id], :]
        rangle_err = error_rates[neurons_ids[id], :]

        ax.plot(np.arange(8), rangle,  lw=1, color=cr.pal_extended[c+3])
        ax.plot(np.arange(8), rangle,  lw=1, color=cr.pal_extended[c+3], ls='none', marker='o', ms=cr.ms)
        ax.errorbar(np.arange(8), rangle, yerr=rangle_err,  color=cr.pal_extended[c+3], fmt='none') 
        ax.set_xticks([0, 4, 8], ['0', 'π/2', 'π'])
        ax.set_xlabel("θ")
        ax.set_ylabel("Rate")


def plot_tuning_curve(ax, units, rates):
    for layer in ['L23', 'L4']:
        neurons_layer = fl.filter_neurons(units, layer=layer, cell_type='exc')
        rates_layer = rates[neurons_layer['id'], :]
    
        #Mean and its standard error
        tcurve     = np.mean(utl.shift_multi(rates_layer, neurons_layer['pref_ori']), axis=0) 
        tcurve_err = np.std(utl.shift_multi(rates_layer, neurons_layer['pref_ori']), axis=0) / np.sqrt(rates_layer.shape[0])

        tcurve     = plotutils.shift(tcurve)
        tcurve_err = plotutils.shift(tcurve_err)
        ax.fill_between(np.arange(9), tcurve - tcurve_err, tcurve + tcurve_err, color=cr.lcolor[layer], alpha=0.5, edgecolor=None)
        ax.plot(np.arange(9), tcurve, color=cr.lcolor[layer], label=layer)
        ax.plot(np.arange(9), tcurve, color=cr.dotcolor[layer], ls="none", marker='o', ms=cr.ms)

    ax.set_xticks([0, 4, 8], ['-π/2', '0', 'π/2'])
    ax.set_xlabel("Δθ")
    ax.set_ylabel("r(Δθ)")
    ax.legend(loc='best')

    return 



def fraction_tuned(ax, data, fstitle=8):
    barw = 0.1
    ybars = [0, barw] 
    offset = 0.02 #To display text

    #Create a Pandas Series which contains the number of tuned neurons in a layer
    #The value is accesed by the key of the pandas dataframe, e.g. n_tuned["L2/3"]
    #tuned_neurons = utl.get_tuned_neurons(data)
    tuned_neurons = fl.filter_neurons(data, tuning="tuned") 
    n_tuned = tuned_neurons.groupby("layer").size()
    total_neurons = data.groupby("layer").size() 

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
    ax.set_xlabel("% selective neurons", labelpad=-2., fontsize=fstitle)





def plot_resultant_dist(ax, v1_neurons, rates):
    bins = np.linspace(0, 1, 30)
    bins_centered = 0.5*(bins[1:] + bins[:-1])
    for layer in ['L23', 'L4']:
        tuned_neurons = fl.filter_neurons(v1_neurons, tuning="tuned", layer=layer)
        ratestuned = rates[tuned_neurons['id']]
        cvo, cvd = mutl.compute_circular_variance(ratestuned, orionly=True)

        #Histogram them, normalizing to count (not by density) 
        hist, _ = np.histogram(cvd, bins, density=False, weights=np.ones(len(cvd))/len(cvd))
        ax.step(bins_centered, hist, color = cr.lcolor[layer])

    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Circ. Var")
    ax.set_ylabel("Fract. neurons")














#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()




def plot_figure(figname):
    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.4), layout='constrained')

    #Bottom part of figure has two sketches 
    subfigs = fig.subfigures(nrows=2, height_ratios=[0.65,1])#[1/3, 2/3]) 
    axes = {}
    sketches = subfigs[1].subplots(ncols=2, width_ratios=[0.625, 1.])
    axes['D'] = sketches[0]
    axes['E'] = sketches[1]

    #Top part: example of the tuning currents with the gratings, then tuning curve, then distribution 
    subfig_graphs = subfigs[0].subfigures(ncols=3)
    subfigs_example = subfig_graphs[0].subplots(nrows=1, ncols=1)
    subfigs_tcurve = subfig_graphs[1].subplots(nrows=1, ncols=1) #single one!
    subfigs_tuned = subfig_graphs[2].subplots(nrows=2, height_ratios = [0.35, 1.])

    axes['A'] = subfigs_example
    axes['B'] = subfigs_tcurve
    axes['C1'] = subfigs_tuned[0]
    axes['C2'] = subfigs_tuned[1]

    units, connections, rates, error_rates = loader.load_data(return_error=True)
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)

    example_tuning_curve(axes['A'], units, rates, error_rates)

    plot_tuning_curve(axes['B'], matched_neurons, rates)

    fraction_tuned(axes['C1'], matched_neurons) 
    plot_resultant_dist(axes['C2'], matched_neurons, rates)

    fig.get_layout_engine().set(wspace=1/72, w_pad=0)

    axes2label = [axes[k] for k in ['A', 'B', 'C1', 'D', 'E']]
    label_pos  = 2*[[0.1, 0.9]] + [[-0.3, 0.9]] + 2*[[0.1, 1.]] 
    sty.label_axes(axes2label, label_pos)

    for ax in [axes['D'], axes['E']]:
        ax.set_axis_off()

    fig.savefig(f"{args.save_destination}/{figname[:-3]}_clean.pdf",  bbox_inches="tight")

    show_image(axes['D'], "sketch1.png")
    show_image(axes['E'], "new_neurons2.png")
    #show_image(subfigs_example['U'], "horizontal.png")
    #show_image(subfigs_example['V'], "vertical.png")
    #show_image(subfigs_example['W'], "horizontal.png")

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure('fig1.pdf')
