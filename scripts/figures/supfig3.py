import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import torch
from scipy.stats import skew
from KDEpy import FFTKDE

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.currents as curr
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.utils as dutl

import ccmodels.modelanalysis.utils as utl
import ccmodels.modelanalysis.sbi_utils as msbi


import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils

import ccmodels.utils.angleutils as au

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=100):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[0, "mean"]
        meandata[layer]  = p['mean'].values 

    return meandata

def compute_summary_data(units, connections, rates): 
    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    neurons_L4 = fl.filter_neurons(units, layer='L4', tuning='matched')

    rates23 = rates[neurons_L23['id'], :]
    tcurvedata = np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0)
    means_data = compute_conn_prob(units, connections)

    #Take the tuning curve at three points: start, minimum, and end 
    cvo, cvd = utl.compute_circular_variance(tcurvedata, orionly=True)
    r0 = tcurvedata[0]
    rf = tcurvedata[-1]

    #Connection probability reduction at beginning and end for L23 adn L4
    #pL23 = 0.5 * (means_data['L23'][0] + means_data['L23'][-1])
    pL23 = means_data['L23'][-1]
    pL23mid = means_data['L23'][2]
    #pL4  = 0.5 * (means_data['L4'][0] + means_data['L4'][-1])
    pL4  = means_data['L4'][-1]
    pL4mid  = means_data['L4'][2]

    summary_data = np.zeros(10)
    summary_data[0] = r0
    summary_data[1] = rf
    summary_data[2] = cvd
    summary_data[3] = pL23
    summary_data[4] = pL4
    #summary_data[5] = pL23mid
    #summary_data[6] = pL4mid

    cvoexp, cvdexp = utl.compute_circular_variance(rates23, orionly=True)

    #Started in 5 before
    #summary_data[7] = np.mean(cvdexp)
    #summary_data[8] = np.std(cvdexp)
    #summary_data[9] = skew(cvdexp)

    summary_data[5] = np.mean(cvdexp)
    summary_data[6] = np.std(cvdexp)
    summary_data[7] = skew(cvdexp)

    logrates = np.log(rates23.flatten())
    #summary_data[10] = np.mean(logrates)
    #summary_data[11] = np.std(logrates)
    summary_data[8] = np.mean(logrates)
    summary_data[9] = np.std(logrates)

    return summary_data


def load_rates(inputfolder, simid, nsims):
    rate_id = simid // nsims 
    part_id = simid - rate_id * nsims 
    return np.load(f"data/model/simulations/{inputfolder}/{rate_id}_rates{part_id}.npy")

def compute_errors(summary_data, folder_files, is_sbi):
    if is_sbi:
        nsims = 100 
        sims_per_file = 100
    else:
        nsims = 1000
        #Lets add another 16 summary stats, CV + rate dists
        sims_per_file = 1000
    
    nparams = 8
    nfiles = 100
    params = np.empty((0, nparams))
    summary_stats = np.empty((0,5))

    for i in range(nfiles):
        inputfile = np.loadtxt(f"data/model/simulations/{folder_files}/{i}.txt")
        if inputfile.size > 0:
            params = np.vstack((params, inputfile[:, :nparams]))
            #Observe that even if mode is NOT kin, the indegree is always stored as a parameter (it's just 400) so the summary stats start always from 9 and does not depend on nparameters
            #Take the tuning curve at three points: start, minimum, and end 
            cvo, cvd = utl.compute_circular_variance(inputfile[:, 9:17], orionly=True)
            r0 = inputfile[:, 9]
            rf = inputfile[:, 16]

            pL23 = inputfile[:,21] 
            pL4  = inputfile[:, 26] 
            #pL23mid = inputfile[:,19] 
            #pL4mid  = inputfile[:, 24] 

            #stats = np.vstack((r0, rf, cvd, pL23, pL4, pL23mid, pL4mid)).transpose()
            stats = np.vstack((r0, rf, cvd, pL23, pL4)).transpose()

            summary_stats = np.vstack((summary_stats, stats))


    #Increase the size of the summary stats
    start_ind = 5
    summary_stats = np.hstack((summary_stats, np.zeros((summary_stats.shape[0], 5))))


    for sim in range(sims_per_file * nfiles):
        readrates = load_rates(folder_files, sim, nsims)

        cvo, cvd = utl.compute_circular_variance(readrates, orionly=True)
        #Started on 5!
        summary_stats[sim, start_ind] = np.mean(cvd)
        summary_stats[sim, start_ind+1] = np.std(cvd)
        summary_stats[sim, start_ind+2] = skew(cvd)

        logrates = np.log(readrates.flatten())
        summary_stats[sim, start_ind+3] = np.mean(logrates)
        summary_stats[sim, start_ind+4] = np.std(logrates)

    ndata = params.shape[0]
    errors = np.empty((ndata, 4))

    for i in range(ndata):
        #errors[i,0] = np.sum((summary_stats[i, 0:3] - summary_data[0:3])**2)
        #errors[i,1] = np.sum((summary_stats[i, 3:7] - summary_data[3:7])**2)
        #errors[i,2] = np.sum((summary_stats[i, 7:10] - summary_data[7:10])**2)
        #errors[i,3] = np.sum((summary_stats[i, 10:12] - summary_data[10:12])**2)
        errors[i,0] = np.sum((summary_stats[i, 0:3] - summary_data[0:3])**2)
        errors[i,1] = np.sum((summary_stats[i, 3:5] - summary_data[3:5])**2)
        errors[i,2] = np.sum((summary_stats[i, 5:8] - summary_data[5:8])**2)
        errors[i,3] = np.sum((summary_stats[i, 8:10] - summary_data[8:10])**2)
    
    if is_sbi:
        best = np.argwhere((errors[:,0] < 0.29) & (errors[:,1] < 0.0051) & (errors[:,2] < 0.07) & (errors[:,3] < 2.))[:,0]
        sort_index = errors[best, 3]
        best_error_sort = np.argsort(sort_index)
        best_pars = params[best[best_error_sort], :]
        best_error = errors[best[best_error_sort], :]
        return errors, best_pars[0,:], best_error[0,:] 
    else:
        return errors

def plot_posterior_distrib(axes, posterior_samples, intervals, color, bw='ISJ'):

    #Plot with a different colormap to differentiate from rel error plots
    labels = [r"$J$", r"$g$", r"$\theta$", r"$\sigma$", r"$I_E$", r"$I_I$", r"$\beta_{23}$", r"$\beta_{4}$"]

    #A plot for each parameter
    for param in range(len(axes)):
        x, y = FFTKDE(kernel='gaussian', bw=bw).fit(posterior_samples[:,param].numpy()).evaluate()
        axes[param].plot(x, y, color=color)

        #Despine and clean axes
        axes[param].set_yticks([])
        axes[param].set_ylim(0, 1.1*np.max(y))

        #Set labels
        axes[param].set_xlabel(labels[param], fontsize=14)


    #Finish graph
    return

def plot_sbi_result(axes, sbinet, summary_data, best_pars, color):
    j0, jf = 0., 4.
    g0, gf = 0., 5.
    sigmaE0, sigmaEf = 7., 12.
    sigmaI0, sigmaIf = 7., 12.
    hei0, heif = 50., 150.
    hii0, hiif = 100., 500.
    b230, b23f = 0.1, 0.6 
    b40, b4f   = 0.1, 0.6 

    intervals = [[j0, jf], [g0, gf], [sigmaE0, sigmaEf], [sigmaI0, sigmaIf], [hei0, heif], [hii0, hiif], [b230, b23f], [b40, b4f]]

    posterior = msbi.load_posterior(f"data/model/sbi_networks/{sbinet}.sbi") 
    pars = posterior.sample((100000,), x=summary_data).numpy()
    parameters_sample = torch.tensor(pars)

    ncols = 8
    bw = 'ISJ'
    inferred = msbi.get_estimation_parameters(parameters_sample, ncols, joint_posterior=False)
    plot_posterior_distrib(axes, parameters_sample, intervals, color, bw=bw)

    for j in range(len(axes)):
        axes[j].axvline(best_pars[j], color='black', ls=":")

def plot_errors(axes, errors_random, errors_sbi, best_error, c1):

    c2 = 'gray' 

    bin_vec = [[0, 100, 300], [0, 0.2, 100], [0, 0.5, 100], [0, 2., 100]]
    ylabels = ["tuning curve", "conn. prob.", "circ. var.", "rate dist."]

    for i in range(4):
        binslims = bin_vec[i]
        bins = np.linspace(binslims[0], binslims[1], binslims[2])

        h, edges = np.histogram(errors_random[:,i], bins=bins, density=True)
        edges = 0.5 * (edges[1:] + edges[:-1])
        axes[i].plot(edges, h, color=c2, label='Random')

        h, edges = np.histogram(errors_sbi[:,i], bins=bins, density=True)
        edges = 0.5 * (edges[1:] + edges[:-1])
        axes[i].plot(edges, h, color=c1, label='SBI')

        axes[i].axvline(best_error[i], color='black', ls=":")
        axes[i].set_xlabel(f"Error {ylabels[i]}")

    axes[0].set_ylabel("Prob. density")
    axes[0].legend(loc='best')

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname, is_tuned):
    # load files
    units, connections, rates = loader.load_data(orientation_only=True)
    connections = fl.remove_autapses(connections)

    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    if is_tuned:
        dataname = "tunedk150"
        sbiname = "tunedk150_mini"
    else:
        dataname = "randk150"
        sbiname = "randk150_mini"

    summary_data                       = compute_summary_data(units, connections, rates)
    errors_random                      = compute_errors(summary_data, dataname, False)
    errors_sbi, best_pars, best_error  = compute_errors(summary_data, f"sbi_{dataname}", True)

    if is_tuned:
        best_pars = best_pars = np.array([9.94296014e-01, 2.04172418e-01, 7.58484888e+00, 8.25478363e+00, 6.88277740e+01, 4.54100189e+02, 1.44280359e-01, 1.49676427e-01])
    else:
        best_pars = np.array([7.92474210e-01, 3.59854251e-01, 7.69989061e+00, 7.60365248e+00, 9.98869781e+01, 4.66699005e+02, 2.98812389e-01, 1.46130979e-01])


    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.5), layout="constrained")

    axes = fig.subplot_mosaic(
        """
        1111
        ABCD
        EFGH
        2222
        XYZW
        """, height_ratios=[0.1, 0.5, 0.5, 0.1, 1]
    )

    color = cr.pal_extended[1]

    #Add some titles
    for i in "12":
        axes[i].set_axis_off()
    axes['1'].text(0.5, 0.5, "Parameter posterior density distributions from SBI", weight='bold', horizontalalignment='center')
    axes['2'].text(0.5, 0.5, "Errors in summary statistics", weight='bold', horizontalalignment='center')

    plot_sbi_result([axes[k] for k in 'ABCDEFGH'], sbiname, summary_data, best_pars, color)
    plot_errors([axes[k] for k in 'XYZW'], errors_random, errors_sbi, best_error, color)


    axes2label = [axes[k] for k in list('ABCDEFGHXYZW')]
    label_pos  = [[0.05, 0.95]] * 8 + [[0.1, 0.95]] * 4 
    sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure("supfig3_normal.pdf", is_tuned=False)
plot_figure("supfig3_tuned.pdf",  is_tuned=True)