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
def diff_emergent2target_prefori(ax, diff_ori, color, label):


    bins = np.arange(-4.5, 5.5)

    hist, edges = np.histogram(diff_ori, bins=bins)

    hist[0] = hist[-1] #Boundary conditions for angle

    hist = hist / len(diff_ori)
    lines, = ax.plot(bins[:-1]+0.5, hist, marker='.', color=color, label=label)

    ax.set_xlabel(r"$\hat \theta _\text{targt}- \hat \theta _\text{emerg}$")
    ax.set_ylabel('Frac. of Neurons')
    ax.set_xticks([-4, 0, 4], ['-π/2', '0', 'π/2'])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylim(0., 0.41)
    
    return lines

def plot_ratedist(ax, re, color):
    bins = np.linspace(0.01, 25, 60)
    w = np.ones(re.size) / re.size

    ax.hist(re.ravel(), density=False,  weights=w, histtype='step', bins=bins, color=color)

    ax.set_xlabel("Rate (spk/s)")
    ax.set_ylabel('Fract. of neurons')
    #ax.set_xscale('log')
    return


#def circular_variance(ax, re, color):
def circular_variance(ax, cved, color):
    bins = np.linspace(0,1,50)

    w = np.ones(cved.size) / cved.size
    ax.hist(cved, bins=bins, density=False, weights=w, color=color, histtype='step')

    ax.set_xlabel("Circ. Var.")
    ax.set_ylabel("Fract. of neurons")


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


def conn_prob_osi(axL23, axL4, meandata, error, colorL23, colorL4, half=True):

    #Plot it!
    angles = np.linspace(0, np.pi/2, 5)
    axes = {'L23': axL23, 'L4': axL4}
    colors = {'L23': colorL23, 'L4': colorL23}
    plots = {'L23':None, 'L4':None}

    for layer in ["L23", "L4"]:
        low_band  = meandata[layer] - error[layer]
        high_band = meandata[layer] + error[layer]

        axes[layer].fill_between(angles, low_band, high_band, color = colors[layer], alpha = 0.2)
        axes[layer].plot(angles, meandata[layer], color = colors[layer])

        #Then just adjust axes and put a legend
        axes[layer].tick_params(axis='both', which='major')
        axes[layer].set_xlabel(r"$|\hat \theta _\text{post} - \hat \theta _\text{pre} |$")
        #axes[layer].set_ylabel(r"$p(\Delta \theta) / p(0)$")
        axes[layer].set_ylabel("Conn. Prob. \n(Normalized)")

        axes[layer].set_ylim(0.5, 1.1)

        axes[layer].set_xticks([0, np.pi/4, np.pi/2], ["0", "π/4", "π/2"])
    
    return 
def make_bar_plot(ax, cvsims, cvdata, title):

    print(cvsims.shape)
    m = cvsims.mean(axis=0)
    s = cvsims.std(axis=0) / np.sqrt(len(cvsims))

    x = np.arange(4)

    print(m.shape)
    ax.bar(x, m, color = cr.reshuf_color, edgecolor='k') 
    ax.errorbar(x, m, yerr = s, color = 'black', marker = 'none', ls='none') 

    ax.axhline(cvdata, ls='--', color = 'black')
    ax.text(x[1], cvdata + 0.005, "Experiment")

    ax.tick_params(axis='x', labelrotation=20)
    ax.set_xticks(x, ['Original', 'All Reshf.', 'L23 Reshf.', 'L4 Reshf.'])
    ax.set_ylabel("Circ. Var.")
    ax.set_ylim(0, 0.4)

    ax.set_title(title)
    return

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 5''')

# Adding and parsing arguments
#parser.add_argument('datafolder', type=str, help='Place where the circular variances are saved')
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname, is_tuned=True, generate_data=True):

    if is_tuned:
        figname += 'tuned'
        filename = 'v1300_def_tuned'
        #cvcomp = np.loadtxt(f"data/model/simulations/{args.datafolder}/cvtuned.txt")
    else:
        figname  += 'normal'
        filename = 'v1300_def'
        #cvcomp = np.loadtxt(f"data/model/simulations/{args.datafolder}/cvnormal.txt")

    nexp = 10 

    # load files
    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)


    #filename = 'best_ale'
    sty.master_format()
    fig, axes = plt.subplot_mosaic(
    """
    ABC
    DEL
    """,
    figsize=sty.two_col_size(height=9.5), layout='constrained') 

    colors = cr.reshuf_color 
    labels = ['Original', 'Reshuffled', 'L23 reshuffled', 'L4 reshuffled']
    #labels = ['Original', 'All reshfl.', 'L23 reshfl.', 'L4 reshfl.']
    legend_handles = []

    cvcomp = np.empty((0, 4))

    for i, reshuffle_mode in enumerate(['', 'all', 'L23', 'L4']):
    #for i, reshuffle_mode in enumerate(['', 'all']):

        c23 = colors[i]
        label = labels[i]

        if generate_data:

            diff_ori = np.empty(0)
            allrates = np.empty(0)
            allcircv = np.empty(0)
            probmean = {'L23' : np.zeros(5), 'L4' : np.zeros(5)} 
            proberr = {'L23' : np.zeros(5), 'L4' : np.zeros(5)} 

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
                target_ori = target_ori[:ne]
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
            np.save(f"{args.save_destination}/{figname}_{i}_probmeanL4", probmean['L4'])
            np.save(f"{args.save_destination}/{figname}_{i}_proberroL4", proberr['L4'])

        else:
            probmean = {}
            proberr  = {}
            currmean = {}
            currerr  = {}

            diff_ori = np.load(f"{args.save_destination}/{figname}_{i}_angl_data.npy")
            allrates = np.load(f"{args.save_destination}/{figname}_{i}_rate_data.npy")
            allcircv = np.load(f"{args.save_destination}/{figname}_{i}_circ_data.npy")
            probmean['L23'] = np.load(f"{args.save_destination}/{figname}_{i}_probmeanL23.npy")
            proberr['L23']  = np.load(f"{args.save_destination}/{figname}_{i}_proberroL23.npy")
            probmean['L4']  = np.load(f"{args.save_destination}/{figname}_{i}_probmeanL4.npy")
            proberr['L4']   = np.load(f"{args.save_destination}/{figname}_{i}_proberroL4.npy")


        #First time we need to resize this to the number of Exc neurons in the simulation, which was not known a priori
        if reshuffle_mode == '':
            cvcomp.resize((allcircv.shape[0], 4))
        
        cvcomp[:, i] = allcircv

        #diff_emergent2target_prefori(axes['A'], exc_pref_ori, target_ori, c23)    
        handle = diff_emergent2target_prefori(axes['A'], diff_ori, c23, label)    
        legend_handles.append(handle)

        #plot_ratedist(axes['B'], re, c23)
        plot_ratedist(axes['B'], allrates, c23)

        #circular_variance(axes['C'], re, c23)
        circular_variance(axes['C'], allcircv, c23)

        print("all ", allcircv.shape)
        print("comp ", cvcomp.shape)
        #cvcomp = np.vstack((cvcomp, allcircv))

        #p1, p2 = conn_prob_osi(axes['D'], axes['E'], units_sample, connections_sample, c23, c4)
        conn_prob_osi(axes['D'], axes['E'], probmean, proberr, c23, c23)


    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    _, cv_data = utl.compute_circular_variance(rates[units_e['id']], orionly=True)
    aver_cv = cv_data.mean()

    #axes['L'].set_axis_off()
    make_bar_plot(axes['L'], cvcomp, aver_cv, "")
    axes['A'].legend(handles=legend_handles, loc=(0.1, 0.55))


    axes2label = [axes[key] for key in 'ABCDEL']
    label_pos  = [-0.25, 1.05] * 6 
    sty.label_axes(axes2label, label_pos)
    

    fig.savefig(f"{args.save_destination}/{figname}.pdf",  bbox_inches="tight")

plot_figure("fig5", is_tuned=False,  generate_data=True)
plot_figure("fig5", is_tuned=True,  generate_data=True)