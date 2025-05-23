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

from sklearn.preprocessing import robust_scale

# ---------------- Aux functions for computations -----------------

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=1000):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=n_samps)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]
        #meandata[layer]  = plotutils.add_symmetric_angle(p['mean'].values)
        meandata[layer]  = p['mean'].values

    return meandata

def compute_conn_prob_witherr(v1_neurons, v1_connections, half=True, n_samps=1000):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=n_samps)
    meandata = {}
    stddata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]
        meandata[layer]  = plotutils.add_symmetric_angle(p['mean'].values)
        stddata[layer]  = plotutils.add_symmetric_angle(p['std'].values)

    return meandata, stddata 


def normalize_errors(*arrays):
    """Apply robust scaling (median-centered, MAD-scaled) to each error array."""
    return [robust_scale(arr.reshape(-1, 1)).flatten() for arr in arrays]

def compute_total_loss(error_arrays, weights=None):
    """Compute a weighted total loss from a list of normalized error arrays."""
    error_matrix = np.vstack(error_arrays).T
    if weights is None:
        weights = np.ones(error_matrix.shape[1])
    return error_matrix @ weights

def load_rates(simid, ratefolder, is_sbi = True):
    if is_sbi:
        rate_id = simid // 100
        part_id = simid - rate_id * 100
        return np.load(f"{ratefolder}/{rate_id}_rates{part_id}.npy")
    else:
        rate_id = simid // 1000
        part_id = simid - rate_id * 1000
        return np.load(f"{ratefolder}/{rate_id}_rates{part_id}.npy")


def load_observable_errors(nsims, datafolder, summary_data, ntops, is_sbi = True): 
    # Process rate data and compute distances
    sampled_rate_histograms = []
    sampled_cv_histograms = []

    rate_dists_L1 = []
    cv_dists_L1 = []

    nsims = 100
    nparams = 8
    params = np.empty((0, nparams))
    summary_stats = np.empty((0,24)) 

    for i in range(nsims):
        inputfile = np.loadtxt(f"{datafolder}/{i}.txt")
        if inputfile.size > 0:
            params = np.vstack((params, inputfile[:, :nparams]))
            summary_stats = np.vstack((summary_stats, inputfile[:, 9:]))
    ndata = params.shape[0]

    rate_bins = np.linspace(0, 10, 20)  
    cv_bins = np.linspace(0, 1, 20)    

    avg_tuning_curves = [] 
    # Loop over each simulation to extract the rate data and compute histograms
    for sim_idx in range(ndata):
        sampled_rates_here = load_rates(sim_idx, datafolder, is_sbi)
        
        # Compute circular variance (CV)
        _, cvd = utl.compute_circular_variance(sampled_rates_here, orionly=True)

        # Compute rate histogram
        rate_hist, _ = np.histogram(sampled_rates_here.ravel(), bins=rate_bins, density=True)
        sampled_rate_histograms.append(rate_hist)

        # Compute CV histogram
        cv_hist, _ = np.histogram(cvd, bins=cv_bins, density=True)
        sampled_cv_histograms.append(cv_hist)

        # Compute L1 mean distances to data histograms 
        rate_dist = np.abs(rate_hist - summary_data['ratehist']).mean()
        cv_dist = np.abs(cv_hist - summary_data['cvhist']).mean()
        rate_dists_L1.append(rate_dist)
        cv_dists_L1.append(cv_dist)
        
        # Compute average tuning curve after shifting each neuron's curve
        avg_tuning_curve = np.mean(dutl.shift_multi(sampled_rates_here, np.argmax(sampled_rates_here, axis=1)), axis=0)
        avg_tuning_curves.append(avg_tuning_curve)


    # Convert the lists to arrays for further analysis
    sampled_rate_histograms = np.array(sampled_rate_histograms)
    sampled_cv_histograms = np.array(sampled_cv_histograms)
    tcurve_sim=summary_stats[:, :8]
    P_conn23_sim=summary_stats[:, 8:16]
    P_conn4_sim=summary_stats[:, 16:]


    rate_dists_L1 = np.array(rate_dists_L1)
    cv_dists_L1 = np.array(cv_dists_L1)
    avg_tuning_curves_L1 = np.mean(np.abs(tcurve_sim- summary_data['tcurve']),axis=1)
    avg_P_conn23_L1 = np.mean(np.abs(P_conn23_sim- summary_data['pL23']),axis=1)
    avg_P_conn4_L1 = np.mean(np.abs(P_conn4_sim- summary_data['pL4']),axis=1)

    # Normalize
    norm_errors = normalize_errors(avg_tuning_curves_L1, 
                                rate_dists_L1,
                                cv_dists_L1, 
                                avg_P_conn23_L1, 
                                avg_P_conn4_L1)
    labels = ["Tuning", "Rate","CV", "Conn L2/3", "Conn L4", ]

    # Compute total loss
    weights = np.array([10, 1., 2, 1, 1])

    total_loss = compute_total_loss(norm_errors, weights=weights)

    # Sort
    sorted_idx = np.argsort(total_loss)
    top_sims = sorted_idx[:ntops]
    print("BEST PARS!")
    print(params[sorted_idx[0]])

    summary_sims = {}
    summary_sims['tcurve'] = tcurve_sim[top_sims]
    summary_sims['pL23']   = P_conn23_sim[top_sims]
    summary_sims['pL4']    = P_conn4_sim[top_sims] 
    summary_sims['cvhist'] = sampled_cv_histograms[top_sims] 
    summary_sims['ratehist'] = sampled_rate_histograms[top_sims]

    return summary_sims 



# ------------------ Plotting functinos -------------------------------------# 

def make_plot(ax, x, data, sims, xlabel, ylabel, title):
    top_n = sims.shape[0] 

    for id in range(top_n):
        ax.plot(x, sims[id], color='gray', alpha = 0.5) 
    ax.plot(x, data, marker='o', color='k')


    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return


# ---------------------------- Figure code --------------------------------

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

    #Get all the data and store in a dict
    summary_data = {}
    rate_bins = np.linspace(0, 10, 20)  
    cv_bins = np.linspace(0, 1, 20)    
    rate_bin_centers = 0.5 * (rate_bins[1:] + rate_bins[:-1])
    cv_bin_centers = 0.5 * (cv_bins[1:] + cv_bins[:-1])


    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    rates23 = rates[neurons_L23['id'], :]
    summary_data['tcurve'] = np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0)
    summary_data['ratehist'], _ = np.histogram(rates23.ravel(), bins=rate_bins, density=True)
    _, cvdexp = utl.compute_circular_variance(rates, orionly=True)
    summary_data['cvhist'], _ = np.histogram(cvdexp, bins=cv_bins, density=True)


    means_data = compute_conn_prob(units, connections)
    summary_data['pL23'] = means_data['L23']
    summary_data['pL4']  = means_data['L4']


    #Read the simulation files and store it in a dict similar to the data one
    summary_sims = load_observable_errors(100, "data/model/simulations/sbi_tunedlowkMINI", summary_data, 100)
                                                                                                                   

    #Do the figure
    sty.master_format()
    fig, axes = plt.subplots(figsize=sty.two_col_size(height=9.5), ncols=3, nrows=2, layout="constrained")

    defaultx = np.arange(0,8) 
    angles = np.arange(1,9)    

    make_plot(axes[0,0], defaultx, summary_data['tcurve'], summary_sims['tcurve'], r'$\hat \theta_\text{post} - \theta$', 'Mean response', 'Tuning Curve')
    make_plot(axes[0,1], rate_bin_centers, summary_data['ratehist'], summary_sims['ratehist'], 'Rate (Hz)', 'Density', 'Rate distribution')
    make_plot(axes[0,2], cv_bin_centers, summary_data['cvhist'], summary_sims['cvhist'], 'Circ. Var.', 'Density', 'Circular Variance')
    make_plot(axes[1,0], angles, summary_data['pL23'], summary_sims['pL23'], r'$\hat \theta_\text{post} - \hat \theta_\text{pre}$',"p(∆θ)", "P. Conn. L23")
    make_plot(axes[1,1], angles, summary_data['pL4'], summary_sims['pL4'], r'$\hat \theta_\text{post} - \hat \theta_\text{pre}$',"p(∆θ)", "P. Conn. L4")

    axes[1,2].set_axis_off()

    #Label the axes
    axes2label = [axes[0,k] for k in range(3)] + [axes[1,k] for k in range(2)]
    label_pos  = [0.8, 0.9] * 5 
    sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}.pdf",  bbox_inches="tight")


plot_figure("testtuned")