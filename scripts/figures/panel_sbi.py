import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.modelanalysis.model as md
import ccmodels.modelanalysis.sbi_utils as msbi
import ccmodels.modelanalysis.currents as mcur
import ccmodels.modelanalysis.utils as utl

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl


import ccmodels.plotting.sbiplots as sbiplot
import ccmodels.plotting.utils as plotutils
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.styles as sty


disorder_average = False 
network = "log5pars"
mode='normal'

ncols = 4 

bw = 'silverman'
sty.master_format()
fig = plt.figure(layout='constrained', figsize=(10, 20))
subfigs = fig.subfigures(4, 1, hspace=0.1, height_ratios = [0.5, 1., 1., 1.])

if network == "5pars":
    features = ['mean_re', 'std_re', 'mean_cve_dir', 'std_cve_dir', 'indiv_traj_std']
elif network == "log5pars":
    features = ['logmean_re', 'logstd_re', 'mean_cve_dir', 'std_cve_dir', 'indiv_traj_std']
elif network == "3pars":
    features = ['mean_re', 'std_re', 'indiv_traj_std']
elif network == "2pars":
    features = ['mean_re', 'std_re']
elif network == "log3pars":
    features = ['logmean_re', 'logstd_re', 'indiv_traj_std']
elif network == "allpars":
    features = ['mean_re', 'std_re', 'mean_cve_dir', 'std_cve_dir', 'cv_curl23', 'cv_curl4', 'indiv_traj_std']

posterior_path = "data/model/sbi_networks/std_5parslog.pkl"
output_path = "scripts/figures/output/aux"
sims_path = "data/model/standard_net"


# ---- Load data

units, connections, rates = loader.load_data(prepath='data', orientation_only=True)
connections = fl.remove_autapses(connections)

connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()


# ---- SBI panel

print("Perform and plot sampling from SBI's trained posterior.")


axes = subfigs[0].subplots(nrows=1, ncols=ncols)
subfigs[0].suptitle("SBI fit")

summary_data = msbi.get_data_summarystats(features)



#Setup the prior
j0, jf = 0, 5
g0, gf = 0., 5.
theta0, thetaf = 10, 25
sigma0, sigmaf = 1, 15 
prior, intervals = msbi.setup_prior(j0, jf, g0, gf, theta0, thetaf, sigma0, sigmaf)

#Load the pretrained network from a file
posterior = msbi.load_posterior(f"{posterior_path}")
posterior_samples = posterior.sample((1000000,), x=summary_data.float())

#Estimate the best parameters
inferred = msbi.get_estimation_parameters(posterior_samples, ncols, joint_posterior=False)
inferred_joint = msbi.get_estimation_parameters(posterior_samples, ncols, joint_posterior=True)
sbiplot.plot_posterior_distrib(axes, posterior_samples, intervals, inferred, bw=bw)
for i in range(ncols):
    axes[i].axvline(inferred_joint[i], color="gray", ls=":")


params = np.empty((0,4)) 
summary_stats = np.empty((0, len(features))) 

onlyfiles = [f for f in listdir(sims_path) if isfile(join(sims_path, f))]
for file in onlyfiles:
    filecontent = pd.read_csv(f'{sims_path}/{file}')

    p = filecontent[['J', 'g', 'theta', 'sigma']]
    #sumstats = filecontent[['mean_re', 'std_re', 'mean_cve_dir', 'std_cve_dir', 'indiv_traj_std']]
    sumstats = filecontent[features]

    if disorder_average:
        p        = p.groupby(np.arange(len(p))//10).mean()
        summary_stats= summary_stats.groupby(np.arange(len(summary_stats))//10).mean()
    
    params = np.vstack([params, p])
    summary_stats = np.vstack([summary_stats, sumstats]) 

imin = np.argmin(np.sum((summary_stats - summary_data.numpy())**2, axis=1))
indiv_pars = params[imin]



# --- Do a simulation with parameters-

print("Do a simulation at most probable parameters.")

orionly = True
local_connectivity = False
formatnumpy = np.vectorize(np.format_float_positional)

if disorder_average:

    nrepetitions = 10 

    for ax_ind, (pars, title) in enumerate(zip([inferred, inferred_joint, indiv_pars], ["Simulation most probable", "Simulation max joint posterior", "Best individual pars"])):
        dist_re = np.empty(0) 
        dist_ri = np.empty(0)
        dist_osi_e = np.empty(0) 
        dist_osi_i = np.empty(0) 
        dist_cv_e = np.empty(0) 
        dist_cv_i = np.empty(0) 

        mean_curr_matched = {'L23':np.zeros(9), 'L4':np.zeros(9), 'Total':np.zeros(9)}
        mean_curr_tuned = {'L23':np.zeros(9), 'L4':np.zeros(9), 'Total':np.zeros(9)}

        for rep in range(nrepetitions):
            #aE_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(400, 8000, pars[0], pars[1],  theta=pars[2], sigma_t=pars[3], local_connectivity=local_connectivity, orionly=orionly, mode=mode, prepath='data')

            aE_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, 400, 8000, pars[0], pars[1],  theta=pars[2]+1, sigma_t=pars[3], 
                                                                                                                                    mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath='data')



            units_sample = units_sample.rename(columns={'pt_root_id':'id'})
            connections_sample = connections_sample.rename(columns={'pre_pt_root_id':'pre_id', 'post_pt_root_id':'post_id'})

            dist_re = np.concatenate([dist_re, re.ravel()])
            dist_ri = np.concatenate([dist_ri, ri.ravel()])

            OSI_E = utl.compute_orientation_selectivity_index(re)    
            OSI_I = utl.compute_orientation_selectivity_index(ri)    

            dist_osi_e = np.concatenate([dist_osi_e, OSI_E])
            dist_osi_i = np.concatenate([dist_osi_i, OSI_I])

            cveo, cved = utl.compute_circular_variance(re, orionly=True)    
            cvio, cvid = utl.compute_circular_variance(ri, orionly=True)    

            dist_cv_e  = np.concatenate([dist_cv_e, cved])
            dist_cv_i  = np.concatenate([dist_cv_i, cvid])

            rates_sample = np.vstack([re, ri, rx])
            currents_matched = mcur.bootstrap_mean_current(units_sample, QJ, rates_sample, tuning=['matched', 'matched'], cell_type=['exc', 'exc'])
            currents_tuned = mcur.bootstrap_mean_current(units_sample, QJ, rates_sample, tuning=['tuned', 'tuned'], cell_type=['exc', 'exc'])

            for layer in ['L23', 'L4', 'Total']:
                mean = plotutils.shift(currents_matched[layer].mean(axis=0))
                mean_curr_matched[layer] += mean

                mean = plotutils.shift(currents_tuned[layer].mean(axis=0))
                mean_curr_tuned[layer] += mean

        for layer in ['L23', 'L4']:
            mean_curr_matched[layer] /= nrepetitions 
            mean_curr_tuned[layer] /= nrepetitions 

        mean_curr_matched['Total'] = mean_curr_matched['L23'] + mean_curr_matched['L4']
        mean_curr_tuned['Total'] = mean_curr_tuned['L23'] + mean_curr_tuned['L4']

        totalmaxmatched = mean_curr_matched['Total'].max()
        totalmaxtuned = mean_curr_tuned['Total'].max()

        for layer in ['L23', 'L4', 'Total']:
            mean_curr_matched[layer] /= totalmaxmatched 
            mean_curr_tuned[layer] /= totalmaxtuned 


        #units, connections, rates = loader.load_data(prepath='data', orientation_only=True)
        l23 = units.loc[(units['layer']=='L23')&(units['tuning_type']!='not_matched'), 'id']
        exprates23 = rates[l23, :]
        l4  = units.loc[(units['layer']=='L4')&(units['tuning_type']!='not_matched'), 'id']
        exprates4 = rates[l4, :]


        axes = subfigs[ax_ind+1].subplots(nrows=2, ncols=2)
        subfigs[ax_ind+1].suptitle(f"{title}; {formatnumpy(pars, 2)}")

        #bins = np.linspace(0, 100, 50)
        bins = np.logspace(-2, 2, 50)

        axes[0,0].hist(exprates23.ravel(), density=True,  histtype='step', bins=bins, label='data_e', color=cr.lcolor["L23"])
        axes[0,0].hist(exprates4.ravel(), density=True,  histtype='step', bins=bins, label='exp4', color=cr.lcolor["L4"])
        axes[0,0].hist(dist_re, density=True,  histtype='step', bins=bins, label='model_e', color=cr.lcolor["L23_modelE"])
        axes[0,0].hist(dist_ri, density=True,  histtype='step', bins=bins, label='model_i', color=cr.lcolor["L23_modelI"])
        axes[0,0].legend(loc="best")
        #axes[0,0].set_yscale('log')

        axes[0,0].set_xlabel("r")
        axes[0,0].set_ylabel("p(r)")

        axes[0,0].set_xscale("log")

        # - series
        for i in range(50):
            axes[0,1].plot(aE_t[i, :])

        axes[0,1].set_xlabel("t")
        axes[0,1].set_ylabel("rE(t)")

        # - currents

        print("Compute associated currents and tuning.")

        connections = fl.remove_autapses(connections)
        connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()
        matchedunits = fl.filter_neurons(units, tuning='matched')
        matchedconnections = fl.synapses_by_id(connections, pre_ids=matchedunits['id'], post_ids=matchedunits['id'], who='both')
        vij = loader.get_adjacency_matrix(matchedunits, matchedconnections)
        vij *= inferred[0]

        currents = mcur.bootstrap_mean_current(units, vij, rates)
        totalmean = currents['Total'].mean(axis=0).max()

        for layer in ['L23', 'L4', 'Total']:
            axes[1,0].plot(plotutils.shift(currents[layer].mean(axis=0)/totalmean), color=cr.lcolor[layer], ls="None", marker="o")


        totalmean = currents['Total'].mean(axis=0).max()
        for layer in ['L23', 'L4', 'Total']:
            mean = mean_curr_matched[layer]
            axes[1,0].plot(mean, label=layer, color=cr.lcolor[layer])


        axes[1,0].set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
        axes[1,0].set_xlabel("Δθ")
        axes[1,0].set_ylabel("Current")


        #- tuning

        bins = np.linspace(0,1,50)

        units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
        _, cv_data = utl.compute_circular_variance(rates[units_e['id']], orionly=True)

        #plt.title(f"J={J_values[best_ix[0]-5]:.2f}, g={g_values[best_ix[1]]:.2f}")
        axes[1,1].hist(cv_data, bins=bins, density=True, alpha=0.5, color=cr.lcolor["L23"])
        axes[1,1].hist(dist_cv_e, bins=bins, density=True, alpha=0.5, color=cr.lcolor["L23_modelE"])
        axes[1,1].hist(dist_cv_i, bins=bins, density=True, alpha=0.5, color=cr.lcolor["L23_modelI"])
        axes[1,1].set_xlabel("OSI")
        axes[1,1].set_ylabel("p(OSI)")

    fig.savefig(f"{output_path}_disorderavg.pdf")

else:
    for ax_ind, (pars, title) in enumerate(zip([inferred, inferred_joint, indiv_pars], ["Simulation most probable", "Simulation max joint posterior", "Best individual pars"])):

        #aE_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(400, 8000, pars[0], pars[1],  theta=pars[2], sigma_t=pars[3], local_connectivity=local_connectivity, orionly=orionly, mode=mode, prepath='data')

        aE_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, 400, 8000, pars[0], pars[1],  theta=pars[2]+1, sigma_t=pars[3], 
                                                                                                                                    mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath='data')



        units_sample = units_sample.rename(columns={'pt_root_id':'id'})
        connections_sample = connections_sample.rename(columns={'pre_pt_root_id':'pre_id', 'post_pt_root_id':'post_id'})

        #units, connections, rates = loader.load_data(prepath='data', orientation_only=True)
        l23 = units.loc[(units['layer']=='L23')&(units['tuning_type']!='not_matched'), 'id']
        exprates23 = rates[l23, :]
        l4  = units.loc[(units['layer']=='L4')&(units['tuning_type']!='not_matched'), 'id']
        exprates4 = rates[l4, :]



        axes = subfigs[ax_ind+1].subplots(nrows=2, ncols=2)
        subfigs[ax_ind+1].suptitle(f"{title}; {formatnumpy(pars, 2)}")

        #bins = np.linspace(0, 100, 50)
        bins = np.logspace(-2, 2, 50)

        axes[0,0].hist(exprates23.ravel(), density=True,  histtype='step', bins=bins, label='data_e', color=cr.lcolor["L23"])
        axes[0,0].hist(exprates4.ravel(), density=True,  histtype='step', bins=bins, label='exp4', color=cr.lcolor["L4"])
        axes[0,0].hist(re.ravel(), density=True,  histtype='step', bins=bins, label='model_e', color=cr.lcolor["L23_modelE"])
        axes[0,0].hist(ri.ravel(), density=True,  histtype='step', bins=bins, label='model_i', color=cr.lcolor["L23_modelI"])
        axes[0,0].legend(loc="best")
        #axes[0,0].set_yscale('log')

        print(exprates23.mean(), re.mean())
        print(exprates23.std(), re.std())

        axes[0,0].set_xlabel("r")
        axes[0,0].set_ylabel("p(r)")

        axes[0,0].set_xscale("log")

        # - series
        for i in range(50):
            axes[0,1].plot(aE_t[i, :])

        axes[0,1].set_xlabel("t")
        axes[0,1].set_ylabel("rE(t)")

        # - currents

        print("Compute associated currents and tuning.")

        connections = fl.remove_autapses(connections)
        connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()
        matchedunits = fl.filter_neurons(units, tuning='matched')
        matchedconnections = fl.synapses_by_id(connections, pre_ids=matchedunits['id'], post_ids=matchedunits['id'], who='both')
        vij = loader.get_adjacency_matrix(matchedunits, matchedconnections)
        vij *= inferred[0]

        currents = mcur.bootstrap_mean_current(units, vij, rates)
        totalmean = currents['Total'].mean(axis=0).max()

        for layer in ['L23', 'L4', 'Total']:
            axes[1,0].plot(plotutils.shift(currents[layer].mean(axis=0)/totalmean), color=cr.lcolor[layer], ls="None", marker="o")


        rates_sample = np.vstack([re, ri, rx])
        QJnormal = QJ 
        currents = mcur.bootstrap_mean_current(units_sample, QJnormal, rates_sample, tuning=['matched', 'matched'], cell_type=['exc', 'exc'])

        totalmean = currents['Total'].mean(axis=0).max()
        for layer in ['L23', 'L4', 'Total']:
            mean = plotutils.shift(currents[layer].mean(axis=0)/totalmean)
            std = plotutils.shift(currents[layer].std(axis=0)/totalmean)
            axes[1,0].plot(mean, label=layer, color=cr.lcolor[layer])
            axes[1,0].fill_between(np.arange(9), mean-std, mean+std, alpha=0.2,color=cr.lcolor[layer])


        axes[1,0].set_xticks([0,4,8], ['-π/2', '0', 'π/2'])
        axes[1,0].set_xlabel("Δθ")
        axes[1,0].set_ylabel("Current")


        #- tuning

        bins = np.linspace(0,1,50)

        cveo, cved = utl.compute_circular_variance(re, orionly=orionly)    
        cvio, cvid = utl.compute_circular_variance(ri, orionly=orionly)    

        units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
        _, cv_data = utl.compute_circular_variance(rates[units_e['id']], orionly=True)

        #plt.title(f"J={J_values[best_ix[0]-5]:.2f}, g={g_values[best_ix[1]]:.2f}")
        axes[1,1].hist(cv_data, bins=bins, density=True, alpha=0.5, color=cr.lcolor["L23"])
        axes[1,1].hist(cved, bins=bins, density=True, alpha=0.5, color=cr.lcolor["L23_modelE"])
        axes[1,1].hist(cvid, bins=bins, density=True, alpha=0.5, color=cr.lcolor["L23_modelI"])
        axes[1,1].set_xlabel("OSI")
        axes[1,1].set_ylabel("p(OSI)")

    fig.savefig(f"{output_path}.pdf")

plt.show()