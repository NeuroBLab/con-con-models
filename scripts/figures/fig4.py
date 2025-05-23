import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os 
sys.path.append(os.getcwd())
import argparse

import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.modelanalysis.currents as mcur
import ccmodels.modelanalysis.sbi_utils as msbi

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as dutl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils



def plot_ratedist(ax, rates, re, ri):
    #bins = np.logspace(-2, 2, 50)
    bins = np.linspace(0.01, 10, 50)

    w = np.ones(ri.size) / ri.size
    ax.hist(ri.ravel(),  density=False, weights=w,  histtype='step',  bins=bins, label='Model I', color=cr.lcolor['L23_modelI'])
    w = np.ones(re.size) / re.size
    ax.hist(re.ravel(),  density=False, weights=w,  histtype='step',  bins=bins, label='Model E', color=cr.lcolor['L23'])

    hist, edges = np.histogram(rates.ravel(), density=False, bins=bins)
    edges = 0.5*(edges[1:] + edges[:-1])

    hist = hist / rates.size

    ax.plot(edges[::2], hist[::2], color=cr.lcolor['L23'], marker='o', ls="--", markersize=cr.ms, zorder=3, label='Data E')

    ax.set_xlabel("Rate (Hz)")
    ax.set_ylabel('Fract. of neurons')

    ax.legend()
    return

def compute_tuning_curves(units_sample, rates_sample):

    #Plot the model results first
    neurons_L23 = fl.filter_neurons(units_sample, layer='L23', cell_type='exc')
    rates23 = rates_sample[neurons_L23['id'], :]
    
    return np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']+4), axis=0) 

def plot_tuning_curves(ax, units, rates, tuning_curves, tuning_error):
    angles = np.arange(8)

    #Plot the model results first
    ax.plot(tuning_curves, color=cr.lcolor['L23'] )
    ax.fill_between(angles, tuning_curves - tuning_error, tuning_curves + tuning_error, color = cr.lcolor['L23'], alpha = 0.2)

    #Then get the real data
    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    rates23 = rates[neurons_L23['id'], :]
    #ax.plot(np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0), ls="none", marker='o', color=cr.lcolor['L23'])
    ax.plot(np.arange(0,8), np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']+4), axis=0), color=cr.lcolor['L23'], ls="--", marker='o', markersize=cr.ms, zorder=3)

    #ax.set_xticks([0,4,8], ['0', 'π/2', 'π'])
    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_ylim(0, 4.1)
    ax.set_xlabel(r'$\hat \theta_\text{post} - \theta$')
    ax.set_ylabel('rate')

    return

def circular_variance(ax, units, rates, cved, cvid):
    bins = np.linspace(0,1,50)

    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    _, cv_data = utl.compute_circular_variance(rates[units_e['id']], orionly=True)

    w = np.ones(cved.size) / cved.size
    ax.hist(cved, bins=bins, density=False, weights=w, histtype='step', color=cr.lcolor['L23']) 
    w = np.ones(cvid.size) / cvid.size
    ax.hist(cvid, bins=bins, density=False, weights=w, histtype='step', color=cr.lcolor['L23_modelI'])

    w = np.ones(cv_data.size) / cv_data.size
    hist, edges = np.histogram(cv_data, density=False, weights=w, bins=bins)
    edges = 0.5*(edges[1:] + edges[:-1])

    #ax.plot(edges, hist, color=cr.lcolor['L23'], ls='none', marker='o')
    ax.plot(edges[::2], hist[::2], color=cr.lcolor['L23'], marker='o', markersize=cr.ms, zorder=3, ls="--")


    ax.set_xlabel("Circ. Var.")
    ax.set_ylabel("Frac. of neurons")

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=100):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=n_samps)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]
        meandata[layer]  = plotutils.add_symmetric_angle(p['mean'].values)

    return meandata

def conn_prob_osi(ax, probmean, proberr, layer, half=True):

    #Plot it!
    angles = plotutils.get_angles(kind="centered", half=half)

    low_band  = probmean[layer] - proberr[layer]
    high_band = probmean[layer] + proberr[layer]
    c = cr.lcolor[layer]

    ax.fill_between(angles, low_band, high_band, color = c, alpha = 0.2)
    ax.plot(angles, probmean[layer], color = c, label = layer)
    ax.scatter(angles, probmean[layer], color = cr.mc, s=cr.ms, zorder = 3)
        


    ax.axvline(0, color="gray", ls=":")

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_ylim(0.5, 1.1)

    ax.set_xlabel(r'$\hat \theta_\text{post} - \hat \theta_\text{pre}$')
    ax.set_ylabel("p(∆θ)")


    plotutils.get_xticks(ax, max=np.pi, half=True)

#"""
def conn_prob_osi_data(ax, v1_neurons, v1_connections, layer, half=True, n_samps = 100):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=100)

    #Plot it!
    angles = plotutils.get_angles(kind="centered", half=half)

    p = conprob[layer]
    c = cr.lcolor[layer]

    #Normalize by p(delta=0), which is at index 3
    p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]

    meandata = p['mean']
    errdata =  p['std']

    meandata  = plotutils.add_symmetric_angle(meandata.values)
    errdata   = plotutils.add_symmetric_angle(errdata.values)

    ax.errorbar(angles, meandata, yerr = errdata,  color = c, ls = "--", label = layer, markersize=cr.ms, marker='o')


    ax.axvline(0, color="gray", ls=":")

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_ylim(0.5, 1.1)

    ax.set_xlabel(r'$\hat \theta_\text{post} - \hat \theta_\text{pre}$')
    ax.set_ylabel("p(∆θ)")


    plotutils.get_xticks(ax, max=np.pi, half=True)
#"""

def compute_currents(units_sample, QJ, rates_sample):

    currents = mcur.bootstrap_mean_current(units_sample, QJ, rates_sample, tuning=['matched', 'matched'], cell_type=['exc', 'exc'], proof=[None, None])

    totalmean = currents['Total'].mean(axis=0).max()
    currmean = {}
    for layer in ['L23', 'L4', 'Total']:
        curr = plotutils.shift(currents[layer].mean(axis=0)/totalmean)
        currmean[layer] = curr

    return currmean 

def plot_currents(ax, units, vij, rates, currmean, currerr):

    #for layer in ['L23', 'L4', 'Total']:
    for layer in ['L23', 'L4']:
        ax.fill_between(np.arange(9), currmean[layer]-currerr[layer], currmean[layer]+currerr[layer], alpha=0.2, color=cr.lcolor[layer])
        ax.plot(currmean[layer], label=layer, color=cr.lcolor[layer])


    currents = mcur.bootstrap_mean_current(units, vij, rates, ['tuned', 'tuned'])
    totalmean = currents['Total'].mean(axis=0).max()
    for layer in ['L23', 'L4', 'Total']:
        mean = plotutils.shift(currents[layer].mean(axis=0)/totalmean)
        #ax.scatter(np.arange(9), mean, color=cr.lcolor[layer], marker='o', s=cr.ms, zorder=3)
        ax.plot(np.arange(9), mean, color=cr.lcolor[layer], marker='o', ms=cr.ms, ls = "--")

    ax.set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
    ax.set_ylim(0, 1.1)

    ax.set_xlabel(r'$\hat \theta_\text{post} - \theta$')
    ax.set_ylabel("μ(∆θ)")
"""
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
    ax.set_xlabel(r'$\hat \theta_\text{post} - \theta$')
    ax.set_ylabel("μ(∆θ)")
"""

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname, generate_data = False):

    # load files
    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)

    sty.master_format()
    fig, axes = plt.subplots(figsize=sty.two_col_size(height=9.5), ncols=3, nrows=2, layout="constrained")

    if generate_data:

        nexp = 2
        diff_ori = np.empty(0)
        allratesE = np.empty(0)
        allratesI = np.empty(0)
        allcircvE = np.empty(0)
        allcircvI = np.empty(0)
        tuning_curve = np.zeros(8)
        tuning_curve_err = np.zeros(8)
        probmean = {'L23' : np.zeros(9), 'L4' : np.zeros(9)} 
        proberr  = {'L23' : np.zeros(9), 'L4' : np.zeros(9)} 
        currmean = {'L23' : np.zeros(9), 'L4' : np.zeros(9), 'Total' : np.zeros(9)} 
        currerr  = {'L23' : np.zeros(9), 'L4' : np.zeros(9), 'Total' : np.zeros(9)} 

        for j in range(nexp):
            #units_sample, connections_sample, rates_sample, n_neurons, target_ori = utl.load_synthetic_data(f"best_ale_{j}")
            #units_sample, connections_sample, rates_sample, n_neurons, target_ori = utl.load_synthetic_data(f"best_search_{j}")
            units_sample, connections_sample, rates_sample, n_neurons, target_ori = utl.load_synthetic_data(f"definitive_random_{j}")
            QJ = loader.get_adjacency_matrix(units_sample, connections_sample)
            ne, ni, nx = n_neurons

            re = rates_sample[:ne, :]
            ri = rates_sample[ne:ne+ni, :]
            rx = rates_sample[ne+ni:, :]
            
            allratesE = np.concatenate((allratesE, re.ravel()))
            allratesI = np.concatenate((allratesI, ri.ravel()))

            cveo, cved = utl.compute_circular_variance(re, orionly=True)    
            allcircvE = np.concatenate((allcircvE, cved))
            cveo, cved = utl.compute_circular_variance(ri, orionly=True)    
            allcircvI = np.concatenate((allcircvI, cved))

            tunings = compute_tuning_curves(units_sample, rates_sample) 
            tuning_curve += tunings  
            tuning_curve_err += tunings**2 

            means      = compute_conn_prob(units_sample, connections_sample)
            print(means)
            means_curr = compute_currents(units_sample, QJ, rates_sample)
            for layer in ['L23', 'L4']:
                probmean[layer] += means[layer]
                proberr[layer] += means[layer]**2
                currmean[layer] += means_curr[layer]
                currerr[layer] += means_curr[layer]**2

        for layer in ['L23', 'L4']:
            probmean[layer] /= nexp
            proberr[layer] /= nexp
            proberr[layer] -= probmean[layer]**2
            proberr[layer] = np.sqrt(proberr[layer])

            currmean[layer] /= nexp 
            currerr[layer]  /= nexp  
            currerr[layer]  -= currerr[layer]**2
            currerr[layer] = np.sqrt(currerr[layer] / nexp) 

        print("!!")
        print(probmean)

        tuning_curve     /= nexp
        tuning_curve_err /= nexp
        tuning_curve_err -= tuning_curve**2
        tuning_curve_err = np.sqrt(tuning_curve_err / nexp)


        np.save(f"{args.save_destination}/{figname}_rateE_data", allratesE)
        np.save(f"{args.save_destination}/{figname}_rateI_data", allratesI)
        np.save(f"{args.save_destination}/{figname}_circE_data", allcircvE)
        np.save(f"{args.save_destination}/{figname}_circI_data", allcircvI)
        np.save(f"{args.save_destination}/{figname}_tuning_curves", tuning_curve) 
        np.save(f"{args.save_destination}/{figname}_tuning_error", tuning_curve_err) 
        np.save(f"{args.save_destination}/{figname}_probmeanL23", probmean['L23'])
        np.save(f"{args.save_destination}/{figname}_proberroL23", proberr['L23'])
        np.save(f"{args.save_destination}/{figname}_probmeanL4", probmean['L4'])
        np.save(f"{args.save_destination}/{figname}_proberroL4", proberr['L4'])
        np.save(f"{args.save_destination}/{figname}_currmeanL23", currmean['L23'])
        np.save(f"{args.save_destination}/{figname}_currerroL23", currerr['L23'])
        np.save(f"{args.save_destination}/{figname}_currmeanL4",  currmean['L4'])
        np.save(f"{args.save_destination}/{figname}_currerroL4",  currerr['L4'])
        np.save(f"{args.save_destination}/{figname}_currmeanLT",  currmean['Total'])
        np.save(f"{args.save_destination}/{figname}_currerroLT",  currerr['Total'])
    else:
        probmean = {}
        proberr  = {}
        currmean = {}
        currerr  = {}

        allratesE = np.load(f"{args.save_destination}/{figname}_rateE_data.npy")
        allratesI = np.load(f"{args.save_destination}/{figname}_rateI_data.npy")
        allcircvE = np.load(f"{args.save_destination}/{figname}_circE_data.npy")
        allcircvI = np.load(f"{args.save_destination}/{figname}_circI_data.npy")
        tuning_curve = np.load(f"{args.save_destination}/{figname}_tuning_curves.npy")
        tuning_curve_err = np.load(f"{args.save_destination}/{figname}_tuning_error.npy")
        probmean['L23'] = np.load(f"{args.save_destination}/{figname}_probmeanL23.npy")
        proberr['L23'] = np.load(f"{args.save_destination}/{figname}_proberroL23.npy")
        probmean['L4'] = np.load(f"{args.save_destination}/{figname}_probmeanL4.npy")
        proberr['L4'] = np.load(f"{args.save_destination}/{figname}_proberroL4.npy")
        currmean['L23'] = np.load(f"{args.save_destination}/{figname}_currmeanL23.npy")
        currerr['L23'] = np.load(f"{args.save_destination}/{figname}_currerroL23.npy")
        currmean['L4'] = np.load(f"{args.save_destination}/{figname}_currmeanL4.npy")
        currerr['L4'] = np.load(f"{args.save_destination}/{figname}_currerroL4.npy")
        currmean['LT'] = np.load(f"{args.save_destination}/{figname}_currmeanLT.npy")
        currerr['LT'] = np.load(f"{args.save_destination}/{figname}_currerroLT.npy")

    print(probmean)

    #plot_posterior(axes[0,0], "cosine_0402_POST")
    plot_ratedist(axes[0,0], rates, allratesE, allratesI)
    plot_tuning_curves(axes[0,1], units, rates, tuning_curve, tuning_curve_err) 
    circular_variance(axes[0,2], units, rates, allcircvE, allcircvI) 
    plot_currents(axes[1,0], units, vij, rates, currmean, currerr) 
    conn_prob_osi(axes[1,1], probmean, proberr, "L23") 
    conn_prob_osi_data(axes[1,1], units, connections, "L23")
    conn_prob_osi(axes[1,2], probmean, proberr, "L4") 
    conn_prob_osi_data(axes[1,2], units, connections, "L4")

    axes2label = [axes[0,k] for k in range(3)] + [axes[1,k] for k in range(3)]
    label_pos  = [0.8, 0.9] * 6 
    sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}.pdf",  bbox_inches="tight")

numbers = {'J' : 1.5 + np.random.randn(1000), 'g' : 2 + np.random.randn(1000)}
df = pd.DataFrame(data=numbers)
df.to_csv("data/model/placeholder.csv", index=False)


plot_figure("fig4normal", generate_data=True)