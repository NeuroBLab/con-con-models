import numpy as np
import matplotlib.pyplot as plt

import sys
import os 
sys.path.append(os.getcwd())
import argparse

import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.modelanalysis.currents as mcur

import ccmodels.plotting.utils as plotutils

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr




def plot_ratedist(ax, rates, re, ri):
    bins = np.logspace(-2, 2, 50)

    w = np.ones(ri.size) / ri.size
    ax.hist(ri.ravel(),  density=False, weights=w,  histtype='step',  bins=bins, label='Model I', color=cr.lcolor['L23_modelI'])
    w = np.ones(re.size) / re.size
    ax.hist(re.ravel(),  density=False, weights=w,  histtype='step',  bins=bins, label='Model E', color=cr.lcolor['L23'])

    hist, edges = np.histogram(rates.ravel(), density=False, bins=bins)
    edges = 0.5*(edges[1:] + edges[:-1])

    hist = hist / rates.size

    ax.scatter(edges[::2], hist[::2], color=cr.lcolor['L23'], marker='o', s=cr.ms, zorder=3, label='Data E')

    #ax.hist(rates.ravel(),     density=False,  histtype='step',  bins=bins, label='Data E', color=cr.lcolor['L23'])

    ax.set_xlabel("Rate")
    ax.set_ylabel('Fraction')
    ax.set_xscale('log')
    ax.legend()
    return

def plot_tuning_curves(ax, units, rates, units_sample, rates_sample):

    #Plot the model results first
    neurons_L23 = fl.filter_neurons(units_sample, layer='L23',tuning='untuned', cell_type='exc')
    rates23 = rates_sample[neurons_L23['id'], :]
    ax.plot(np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0), color=cr.lcolor['L23'] )

    #Then get the real data
    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    rates23 = rates[neurons_L23['id'], :]
    #ax.plot(np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0), ls="none", marker='o', color=cr.lcolor['L23'])
    ax.scatter(np.arange(0,8), np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0), color=cr.lcolor['L23'], marker='o', s=cr.ms, zorder=3)

    ax.set_xticks([0,4,8], ['0', 'π/2', 'π'])
    ax.set_xlabel('θ')
    ax.set_ylabel('r(θ)')

    return

def circular_variance(ax, units, rates, re, ri):
    bins = np.linspace(0,1,50)

    cveo, cved = utl.compute_circular_variance(re, orionly=True)    
    cvio, cvid = utl.compute_circular_variance(ri, orionly=True)    

    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    _, cv_data = utl.compute_circular_variance(rates[units_e['id']], orionly=True)

    ax.hist(cved, bins=bins, density=True, histtype='step', color=cr.lcolor['L23']) 
    ax.hist(cvid, bins=bins, density=True, histtype='step', color=cr.lcolor['L23_modelI'])

    #ax.hist(cv_data, bins=bins, density=True, alpha=0.5, color=cr.lcolor['L23']) 
    hist, edges = np.histogram(cv_data, density=True, bins=bins)
    edges = 0.5*(edges[1:] + edges[:-1])

    #ax.plot(edges, hist, color=cr.lcolor['L23'], ls='none', marker='o')
    ax.scatter(edges[::2], hist[::2], color=cr.lcolor['L23'], marker='o', s=cr.ms, zorder=3)

    ax.set_xlabel("CirVar")
    ax.set_ylabel("p(CirVar)")

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
        ax.scatter(angles, meandata, color = cr.mc, s=cr.ms, zorder = 3)


    ax.axvline(0, color="gray", ls=":")

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel('∆θ')
    ax.set_ylabel("p")


    plotutils.get_xticks(ax, max=np.pi, half=True)

def plot_currents(ax, units, vij, rates, units_sample, QJ, rates_sample):

    currents = mcur.bootstrap_mean_current(units_sample, QJ, rates_sample, tuning=['matched', 'matched'], cell_type=['exc', 'exc'], proof=[None, None])

    totalmean = currents['Total'].mean(axis=0).max()
    for layer in ['L23', 'L4', 'Total']:
        mean = plotutils.shift(currents[layer].mean(axis=0)/totalmean)
        std = plotutils.shift(currents[layer].std(axis=0)/totalmean)
        ax.plot(mean, label=layer, color=cr.lcolor[layer])
        ax.fill_between(np.arange(9), mean-std, mean+std, alpha=0.2, color=cr.lcolor[layer])


    currents = mcur.bootstrap_mean_current(units, vij, rates, ['tuned', 'tuned'])
    totalmean = currents['Total'].mean(axis=0).max()
    for layer in ['L23', 'L4', 'Total']:
        mean = plotutils.shift(currents[layer].mean(axis=0)/totalmean)
        #std = plotutils.shift(currents[layer].std(axis=0)/totalmean)
        #ax.plot(mean, color=cr.lcolor[layer], ls='none', marker='o')
        ax.scatter(np.arange(9), mean, color=cr.lcolor[layer], marker='o', s=cr.ms, zorder=3)

    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_xlabel('Δθ')
    ax.set_ylabel('μ')


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
    fig, axes = plt.subplots(figsize=sty.two_col_size(height=9.5), ncols=3, nrows=2, layout="constrained")

    units_sample, connections_sample, rates_sample, n_neurons = utl.load_synthetic_data("prueba")
    QJ = loader.get_adjacency_matrix(units_sample, connections_sample)
    ne, ni, nx = n_neurons
    re = rates_sample[:ne, :]
    ri = rates_sample[ne:ne+ni, :]

    plot_ratedist(axes[0,0], rates, re, ri)
    plot_tuning_curves(axes[0,1], units, rates, units_sample, rates_sample)
    circular_variance(axes[0,2], units, rates, re, ri)
    plot_currents(axes[1,1], units, vij, rates, units_sample, QJ, rates_sample)

    conn_prob_osi(axes[1,0], units_sample, connections_sample)

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")

plot_figure("fig4.pdf")