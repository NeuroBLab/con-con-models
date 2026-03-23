import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import ttest_ind_from_stats

import sys
import os 
sys.path.append(os.getcwd())
os.environ['USE_FREQ'] = 'true'
import argparse

import ccmodels.modelanalysis.utils as utl

import ccmodels.utils.distances as au

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.utils as plotutils
import ccmodels.plotting.color_reference as cr

#def diff_emergent2target_prefori(ax, pref_ori, target_ori, color):
def diff_emergent2target_prefori(ax, diff_ori, color, label):


    bins = np.arange(-7.5, 8.5)

    hist, edges = np.histogram(diff_ori, bins=bins)
    hist = hist / hist.sum() 

    xvals = 0.5 * (bins[1:] + bins[:-1]) * 0.02125
    print(xvals)
    print(hist)
    print(hist * xvals**2)
    av  = np.dot(hist,  xvals**2)
    av2 = np.dot(hist,  xvals**4)
    print('mse', av,  np.sqrt(av2 - av**2))
    print(hist.sum() - hist[5])
    print()


    lines, = ax.step(bins[:-1]+0.5, hist, color=color, label=label)

    ax.set_xlabel(r"$\hat k_\text{targt}- \hat k_\text{emerg}$")
    ax.set_ylabel('Neuron frac.')
    ax.set_xticks([-8, 0, 8], ['-0.17', '0.0', '0.17'])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylim(0., 0.41)
    
    return lines

def plot_ratedist(ax, re, color):
    bins = np.linspace(0.01, 25, 60)
    w = np.ones(re.size) / re.size

    print("rate ", re.mean(), re.std())
    ax.hist(re.ravel(), density=False,  weights=w, histtype='step', bins=bins, color=color)

    ax.set_xlabel("Rate")
    #ax.set_ylabel('Neuron frac.')
    return


def spatial_selectivity_index(ax, ssf, color):
    bins = np.linspace(0,1,50)

    w = np.ones(ssf.size) / ssf.size
    ax.hist(ssf, bins=bins, density=False, weights=w, color=color, histtype='step')

    ax.set_xlabel("Circ. Var.")
    #ax.set_ylabel("Neuron frac.")


def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=100):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_prepost(v1_neurons, v1_connections)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        meandata[layer] = p["mean"] / np.max(p["mean"])

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
        #axes[layer].set_ylabel("Conn. Prob. \n(Normalized)")

        axes[layer].set_ylim(0.5, 1.1)

        axes[layer].set_xticks([0, np.pi/4, np.pi/2], ["0", "π/4", "π/2"])
    
    axes['L23'].set_ylabel("Conn. Prob. \n(Normalized)")
    return 

def make_bar_plot(ax, cvsims, cvdata, title):

    m = cvsims.mean(axis=0)
    s = cvsims.std(axis=0) / np.sqrt(len(cvsims))

    x = np.arange(4)

    ax.bar(x, m, color = cr.reshuf_color, edgecolor='k') 
    ax.errorbar(x, m, yerr = s, color = 'black', marker = 'none', ls='none') 

    #ax.axhline(cvdata, ls='--', color = 'gray', lw=1)
    #ax.text(x[1], cvdata + 0.005, "Experiment")

    ax.axhline(m[0], ls='--', color = 'black')

    for i in range(1, 4):
        tstat, pval = ttest_ind_from_stats(m[0], s[0], 10, m[i], s[i], 10, alternative='greater')
        add_sig_bracket(ax, 0, i, m[0], m[i], yerr1=s[0], yerr2=s[1], p=pval, level=i-1, fs=9)


    ax.tick_params(axis='x', labelrotation=20)
    ax.set_xticks(x, ['Original', 'All Reshf.', 'L23 Reshf.', 'L4 Reshf.'])
    ax.set_ylabel("Circ. Var.")
    ax.set_ylim(0,0.6)

    ax.set_title(title)
    return

def p_to_stars(p):
    if p < 1e-3:
        return '***'
    elif p < 1e-2:
        return '**'
    elif p < 5e-2:
        return '*'
    else:
        return 'n.s.'

def add_sig_bracket(ax, x1, x2, y1, y2, yerr1=0.0, yerr2=0.0, p=1.0,
                    level=0, pad_frac=0.03, h_frac=0.1, text_frac=-0.05,fs=11):
    """
    Draw a significance bracket between bars at x1 and x2.
    """
    y0, y1_lim = ax.get_ylim()
    yr = y1_lim - y0

    pad = pad_frac * yr
    h = h_frac * yr
    text_pad = text_frac * yr

    base = max(y1 + yerr1, y2 + yerr2) + pad
    y = base + level * (pad + h + text_pad)

    label = p_to_stars(p)

    ax.plot([x1, x1, x2, x2],
            [y,  y + h, y + h, y],
            color='black', clip_on=False)

    ax.text((x1 + x2) / 2,
            y + h + text_pad,
            label,
            ha='center', va='bottom',
            fontsize=fs)

    return y + h + text_pad

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 5''')

# Adding and parsing arguments
#parser.add_argument('datafolder', type=str, help='Place where the circular variances are saved')
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname, generate_data=True):

    filename = 'v1300_def_spfreq'

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

    ssfcomp = np.empty((0, 4))

    for i, reshuffle_mode in enumerate(['', 'all', 'L23', 'L4']):
    #for i, reshuffle_mode in enumerate(['', 'all']):

        c23 = colors[i]
        label = labels[i]

        if generate_data:

            diff_ori = np.empty(0)
            allrates = np.empty(0)
            allssf = np.empty(0)
            probmean = {'L23' : np.zeros((8,8)), 'L4' : np.zeros((8,8))} 
            proberr = {'L23' : np.zeros((8,8)), 'L4' : np.zeros((8,8))} 

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
                diff_ori = np.concatenate((diff_ori, au.signed_dist_vectorized(target_ori, exc_pref_ori)))

                allrates = np.concatenate((allrates, re.ravel()))

                ssftrial = utl.compute_spatial_selectivity_index(re)
                allssf = np.concatenate((allssf, ssftrial))


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
            np.save(f"{args.save_destination}/{figname}_{i}_ssf_data", allssf)
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
            allssf   = np.load(f"{args.save_destination}/{figname}_{i}_ssf_data.npy")
            probmean['L23'] = np.load(f"{args.save_destination}/{figname}_{i}_probmeanL23.npy")
            proberr['L23']  = np.load(f"{args.save_destination}/{figname}_{i}_proberroL23.npy")
            probmean['L4']  = np.load(f"{args.save_destination}/{figname}_{i}_probmeanL4.npy")
            proberr['L4']   = np.load(f"{args.save_destination}/{figname}_{i}_proberroL4.npy")


        #First time we need to resize this to the number of Exc neurons in the simulation, which was not known a priori
        if reshuffle_mode == '':
            ssfcomp.resize((allssf.shape[0], 4))
        
        ssfcomp[:, i] = allssf

        #diff_emergent2target_prefori(axes['A'], exc_pref_ori, target_ori, c23)    
        handle = diff_emergent2target_prefori(axes['A'], diff_ori, c23, label)    
        legend_handles.append(handle)

        #plot_ratedist(axes['B'], re, c23)
        plot_ratedist(axes['B'], allrates, c23)

        spatial_selectivity_index(axes['C'], allssf, c23)

        #cvcomp = np.vstack((cvcomp, allcircv))

        #TODO will need to be plotted somehow...
        #conn_prob_osi(axes['D'], axes['E'], probmean, proberr, c23, c23)


    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    ssf_data = utl.compute_spatial_selectivity_index(rates[units_e['id']])
    aver_ssf = ssf_data.mean()

    #axes['L'].set_axis_off()
    make_bar_plot(axes['L'], ssfcomp, aver_ssf, "")
    axes['A'].legend(handles=legend_handles, loc=(0.1, 0.55))


    axes2label = [axes[key] for key in 'ABCDEL']
    label_pos  = [-0.25, 1.05] * 6 
    sty.label_axes(axes2label, label_pos)
    

    fig.savefig(f"{args.save_destination}/{figname}.pdf",  bbox_inches="tight")

plot_figure("fig5_sfq",  generate_data=False)