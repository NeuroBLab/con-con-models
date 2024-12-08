import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch as Box
import argparse
from scipy.stats import ttest_ind 

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

def conn_prob_osi(ax, ax_normalized, v1_neurons, v1_connections, half=True):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half)


    #Plot it!
    #angles = plotutils.get_angles(kind="centered", half=half)
    angles = np.arange(9)

    for layer in ["L23", "L4"]:
        p = conprob[layer]
        c = cr.lcolor[layer]

        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] #/p.loc[3, "mean"]

        low_band  = p['mean'] - p['std']
        high_band = p['mean'] + p['std']
        meandata = p['mean']

        low_band  = plotutils.add_symmetric_angle(low_band.values)
        high_band = plotutils.add_symmetric_angle(high_band.values)
        meandata  = plotutils.add_symmetric_angle(meandata.values)

        ax.fill_between(angles, low_band, high_band, color = c, alpha = 0.2)
        ax.plot(angles, meandata, color = c, label = layer)
        ax.scatter(angles, meandata, color = cr.mc, s=cr.ms, zorder = 3)

        #Normalize by p(delta=0), which is at index 3
        low_band  /= p.loc[3, 'mean'] 
        high_band /= p.loc[3, 'mean'] 
        meandata  /= p.loc[3, 'mean'] 

        ax_normalized.fill_between(angles, low_band, high_band, color = c, alpha = 0.2)
        ax_normalized.plot(angles, meandata, color = c, label = layer)
        ax_normalized.scatter(angles, meandata, color = cr.mc, s=cr.ms, zorder = 3)

    ax.axvline(4, color="gray", ls=":")
    ax_normalized.axvline(4, color="gray", ls=":")

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel(r'$\hat \theta_\text{post} - \hat \theta_\text{pre}$')
    ax.set_ylabel("p(∆θ)")

    ax_normalized.tick_params(axis='both', which='major')
    ax_normalized.set_xlabel(r'$\hat \theta_\text{post} - \hat \theta_\text{pre}$')
    ax_normalized.set_ylabel("p(∆θ)/p(0)")

    #plotutils.get_xticks(ax, max=np.pi, half=True)
    #plotutils.get_xticks(ax_normalized, max=np.pi, half=True)
    ax.set_xticks([0, 2, 4, 6, 8], ['-π/2', '', '0', '', 'π/2'])
    ax_normalized.set_xticks([0, 2, 4, 6, 8], ['-π/2', '', '0', '', 'π/2'])

    ax.set_ylim(0, 0.015)

def plot_ratedist(ax, v1_neurons, rates):
    #bins = np.logspace(-2, 2, 40)
    bins = np.linspace(0.01, 10, 50)

    for layer in ['L4', 'L23']: 
        ids = fl.filter_neurons(v1_neurons, tuning='matched', layer=layer).loc[:, 'id'].values
        h, edges = np.histogram(rates[ids, :].ravel(), density=True,  bins=bins)
        dr = edges[1:] - edges[:-1]
        ax.step(edges[:-1], h*dr, color=cr.lcolor[layer], label=layer)

    #ax.set_xscale('log')
    #ax.set_xticks([0.01, 1, 100])
    ax.set_ylabel('Neuron fract.')
    ax.set_xlabel('rate (spk/s)')


def plot_synvoldist(ax, v1_neurons, v1_connections):
    bins = np.logspace(-2, 2, 40)

    for layer in ['L4', 'L23']: 
        conns = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[layer, 'L23'], cell_type=['exc', 'exc'])
        h, edges = np.histogram(conns['syn_volume'].values, density=True, bins=bins)
        dv = edges[1:] - edges[:-1]
        ax.step(edges[:-1], h*dv, color=cr.lcolor[layer])

    ax.set_ylabel('Synap. fract.')
    ax.set_xlabel('Synapse size')
    ax.set_xscale('log')
    ax.set_xticks([0.01, 1, 100])


"""
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
    colors = [cr.pal_extended[i] for i in range(len(labels))]
    for i,r in enumerate([rates.ravel(), tuned_rates, untuned_rates, cotuned_rates, orthogo_rates]):
        h, edges = np.histogram(r, density=True,  bins=bins)
        dr = edges[1:] - edges[:-1]
        ax.step(edges[:-1], h*dr, label=labels[i], color=colors[i])


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
    colors = [cr.pal_extended[i] for i in range(len(labels))]
    for i,r in enumerate([v1_connections['syn_volume'].values, tuned_synvol, untuned_synvol, cotuned_synvol, orthogo_synvol]):
        #h, edges = np.histogram(r, density=True, bins=bins)
        h, edges = np.histogram(r, density=True, bins=bins)
        dv = edges[1:] - edges[:-1]
        ax.step(edges[:-1], h*dv, label=labels[i], color=colors[i])



    ax.set_xscale('log')
"""

#""" v2
def plot_sampling_current(ax, ax_normalized, v1_neurons, v1_connections, rates):
    angles = plotutils.get_angles(kind="centered", half=True)
    nexperiments = 1000 
    frac = 550 / len(v1_connections)

    #Compute the currents in the system
    #mean_cur, std_cur = curr.bootstrap_system_currents(v1_neurons, v1_connections, rates, nexperiments, frac=frac, replace=False)
    mean_cur = curr.bootstrap_system_currents_shuffle(v1_neurons, v1_connections, rates, nexperiments, frac=frac)

    #Total current is shown just in the "unnormalized" version. Also we need to obtain
    #the global total current to normalize according to it
    total_cur = mean_cur['Total'].mean(axis=1)
    norma = np.max(total_cur)
    total_cur = plotutils.shift(total_cur)
    ax.plot(angles, total_cur/norma, label='Total', color=cr.lcolor['Total'])
    ax.scatter(angles, total_cur/norma, color=cr.mc, s=cr.ms, zorder=3)

    #Then show L23 and L4 currents for unnormalized and normalized versions
    for layer in ['L23', 'L4']:
        meancur = mean_cur[layer].mean(axis=1)
        stdcur  = mean_cur[layer].std(axis=1) / np.sqrt(mean_cur[layer].shape[1])

        meancur = plotutils.shift(meancur)
        stdcur = plotutils.shift(stdcur)
        

        ax.fill_between(angles, (meancur-stdcur)/norma, (meancur+stdcur)/norma, color=cr.lcolor[layer], alpha=0.2)
        ax.plot(angles, meancur/norma, label=layer, color=cr.lcolor[layer])
        ax.scatter(angles, meancur/norma, color=cr.mc, s=cr.ms, zorder=3)

        stdcur  /= np.max(meancur)
        meancur /= np.max(meancur)
        ax_normalized.fill_between(angles, meancur-stdcur, meancur+stdcur, color=cr.lcolor[layer], alpha=0.2)
        ax_normalized.plot(angles, meancur, label=layer, color=cr.lcolor[layer])
        ax_normalized.scatter(angles, meancur, color=cr.mc, s=cr.ms, zorder=3)

    plotutils.get_xticks(ax, max=np.pi, half=True)
    plotutils.get_xticks(ax_normalized, max=np.pi, half=True)

    ax.set_xlabel(r'$\hat \theta_\text{post} - \theta$')
    ax_normalized.set_xlabel(r'$\hat \theta_\text{post} - \theta$')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('μ(Δθ)')
    ax_normalized.set_ylabel('μ(Δθ)/μ(0)')
#"""

""" 
def plot_sampling_current(ax, ax_normalized, v1_neurons, v1_connections, rates):
    angles = plotutils.get_angles(kind="centered", half=True)
    nexperiments = 1000 
    frac = 550 / len(v1_connections)

    #Compute the currents in the system
    #mean_cur, std_cur = curr.bootstrap_system_currents(v1_neurons, v1_connections, rates, nexperiments, frac=frac, replace=False)
    mean_cur = curr.bootstrap_system_currents_shuffle(v1_neurons, v1_connections, rates, nexperiments, frac=frac)

    #Total current is shown just in the "unnormalized" version. Also we need to obtain
    #the global total current to normalize according to it
    total_cur = mean_cur['Total'].mean(axis=1)
    norma = np.max(total_cur)
    total_cur = plotutils.shift(total_cur)
    #ax.plot(angles, total_cur/norma, label='Total', color=cr.lcolor['Total'])
    #ax.scatter(angles, total_cur/norma, color=cr.mc, s=cr.ms, zorder=3)


    #Then show L23 and L4 currents for unnormalized and normalized versions
    for layer in ['L23', 'L4']:
        meancur = mean_cur[layer].mean(axis=1)
        stdcur  = mean_cur[layer].std(axis=1) / np.sqrt(mean_cur[layer].shape[1])


        meancur = plotutils.shift(meancur) / norma
        stdcur = plotutils.shift(stdcur) / norma

        ax.fill_between(angles, meancur-stdcur, meancur+stdcur, color=cr.lcolor[layer], alpha=0.2)
        ax.plot(angles, meancur, label=layer, color=cr.lcolor[layer])
        ax.scatter(angles, meancur, color=cr.mc, s=cr.ms, zorder=3)

        mc0 = mean_cur[layer][0, :] / norma 
        mc4 = mean_cur[layer][4, :] / norma 

        test = ttest_ind(mc0, mc4, equal_var=False, alternative='greater') 
        print(test, mean_cur[layer][0, :].shape)
        xpos = np.array([0,1]) + 3*(layer=='L4')
        h = [meancur[4], meancur[8]]
        ax_normalized.bar(xpos, h, color=cr.lcolor[layer])

        offset = 0.2
        sep = 0.05
        ax_normalized.plot([xpos[0], xpos[0], xpos[1], xpos[1]], [h[0]+sep, h[0]+offset, h[0]+offset, h[1]+sep], color='black')
        if test.pvalue > 0.05:
            test_result = " n.s." 
        elif test.pvalue > 0.01:
            test_result = "  *  " 
        elif test.pvalue > 0.001:
            test_result = " * * " 
        else:
            test_result = "* * *" 
        textoffset = [0.02, offset + 0.03]
        ax_normalized.text(xpos[0] + textoffset[0], h[0] +textoffset[1], test_result)

    
    


    plotutils.get_xticks(ax, max=np.pi, half=True)
    ax_normalized.set_xticks([0,1,3,4],['0','π/2','0','π/2'])


    ax.set_xlabel('∆θ')
    ax.set_ylabel('μ(Δθ)')

"""




    
def plot_sampling_current_peaks(ax, v1_neurons, v1_connections, rates):
    frac = 550 / len(v1_connections)

    current = curr.bootstrap_system_currents_peaks(v1_neurons, v1_connections, rates, frac=frac)
    bins = np.arange(-7.5, 8.5, 1)

    for layer in ['L23', 'L4']:
        pref_ori = np.argmax(current[layer], axis=1)
        pref_ori[pref_ori > 3] = pref_ori[pref_ori > 3] - 8 

        hist, _ = np.histogram(pref_ori, bins=bins)
        ax.step(bins[1:], hist, color = cr.lcolor[layer], label=layer)


def tuning_prediction_performance(ax, matched_neurons, matched_connections, rates, nexperiments=1000): 

    angles = np.arange(9)
    tuned_outputs = fl.filter_connections(matched_neurons, matched_connections, tuning="matched", who="pre") 

    prob_pref_ori  = curr.sample_prefori(matched_neurons, tuned_outputs, nexperiments, rates, nsamples=700)
    print(prob_pref_ori)
    print()
    print()
    

    #Plot
    for layer in ['Total', 'L23', 'L4']:
        ax.plot(angles, plotutils.shift(prob_pref_ori[layer]), color=cr.lcolor[layer], label=layer)
        ax.scatter(angles, plotutils.shift(prob_pref_ori[layer]), color=cr.mc, zorder=3, s=cr.ms) 


    ax.set_xlabel(r"$\hat \theta_\text{target} - \hat \theta_\text{emerg}$")
    ax.set_ylabel("Probability")

    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_yticks([0, 0.25, 0.5])


def plot_tuning_curves(ax, units, rates):

    p4legend = [] 

    #Plot the real data
    angles = np.arange(9)
    for layer in ['L23', 'L4']:
        neurons_layer = fl.filter_neurons(units, layer=layer, tuning='matched')
        rateslayer= rates[neurons_layer['id'], :]

        rates_shifted = dutl.shift_multi(rateslayer, neurons_layer['pref_ori'])
        mean_rates = np.mean(rates_shifted, axis=0)
        #std_rates  = np.std(rates_shifted,  axis=0) #/ np.sqrt(rates_shifted.shape[0])

        #ax.fill_between(angles, mean_rates - std_rates, mean_rates + std_rates, color=cr.lcolor[layer], alpha=0.2)
        p, = ax.plot(angles, plotutils.shift(mean_rates), color=cr.lcolor[layer], label=layer) 
        p4legend.append(p)
        ax.scatter(angles, plotutils.shift(mean_rates), color=cr.mc, marker='o', s=cr.ms, zorder=3)

    #ax.set_xticks([0,4,8], ['0', 'π/2', 'π'])
    ax.axvline(4, color='gray', ls=':')
    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_xlabel('θ')
    ax.set_ylabel('r(θ)')

    return p4legend


def plot_figure3(figname):
    # load files
    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    #vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)



    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.9), layout="constrained")
    #ghostax = fig.add_axes([0,0,1,1])
    
    #ghostax.axis('off')

    axes = fig.subplot_mosaic(
        """
        XABL
        YCEF
        WDHG
        """, width_ratios=[0.4, 1, 1, 1]
        #"""
        #ABCD
        #EFGH
        #"""
    )

    plot_ratedist(axes['A'],  matched_neurons, rates)
    p4legend = plot_tuning_curves(axes['B'], units, rates) 
    plot_synvoldist(axes['C'], matched_neurons, matched_connections)

    conn_prob_osi(axes['E'], axes['F'], matched_neurons, matched_connections)
    plot_sampling_current(axes['D'], axes['H'], matched_neurons, matched_connections, rates)
    tuning_prediction_performance(axes['G'], matched_neurons, matched_connections, rates)


    axes2label = [axes[k] for k in 'ABCEFDHG']
    label_pos  = [0.8, 0.9] * 8 
    sty.label_axes(axes2label, label_pos)

    for key in 'XYW':
        axes[key].set_axis_off()

    axes['X'].text(0., 0.5, "Rate\nStatistics", horizontalalignment='center', verticalalignment='center', weight='bold')
    axes['Y'].text(0., 0.5, "Connectivity\nStatistics", horizontalalignment='center', verticalalignment='center', weight='bold')
    axes['W'].text(0., 0.5, "Current\nStatistics", horizontalalignment='center', verticalalignment='center', weight='bold')

    #Legend axis
    axes['L'].axis('off')
    handles, labels = axes['B'].get_legend_handles_labels()
    axes['L'].legend(handles, labels, loc=(0., 0.5), handlelength=1.2)#, ncols=1, loc=(0,0.0), alignment='left')
    #axes['L'].set_axis_off()
    #axes['L'].legend(p4legend,loc='best')

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure3("fig2.pdf")

def plot_figure():
    sty.master_format()

    fig, axes = plt.subplots(ncols=2, figsize=(12,6), layout='constrained')

    units, connections, rates = loader.load_data()
    #import ccmodels.modelanalysis.utils as mutl
    #units, connections, rates = mutl.load_synthetic_data("reshuffled_J4")
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(units, connections)
    vij = vij[matched_neurons['id'], matched_neurons['id']]
    angles = plotutils.get_angles(kind="centered", half=True)

    #conn_prob_osi(ax, matched_neurons, matched_connections)

    #tuning_prediction_performance(ax, matched_neurons, matched_connections, rates)
    #ax.set_ylim(0, 0.5)

    plot_sampling_current(axes[0], axes[1], matched_neurons, matched_connections, rates)


    axes[0].legend()

    fig.savefig(f"curr.pdf",  bbox_inches="tight")

#plot_figure()