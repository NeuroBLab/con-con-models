import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import ttest_ind_from_stats, ttest_ind 

import sys
import os 
sys.path.append(os.getcwd())
os.environ['USE_FREQ'] = 'true'
import argparse

import ccmodels.modelanalysis.utils as utl
import ccmodels.dataanalysis.utils  as dutl

import ccmodels.utils.distances as au

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.utils as plotutils
import ccmodels.plotting.color_reference as cr

def example_tuning_curve(ax, v1_neurons, rates, error_rates, layer='L23'):

    neurons_ids = fl.filter_neurons(v1_neurons, layer=layer, tuning='tuned')
    neurons_ids = neurons_ids['id']

    ids = [8, 3, 11]

    for c,id in enumerate(ids):
        rangle = rates[neurons_ids[id], :]
        rangle_err = error_rates[neurons_ids[id], :]

        ax.plot(np.arange(8), rangle,  lw=1, color=cr.pal_extended[c+3])
        ax.plot(np.arange(8), rangle,  lw=1, color=cr.pal_extended[c+3], ls='none', marker='o', ms=cr.ms)
        ax.errorbar(np.arange(8), rangle, yerr=rangle_err,  color=cr.pal_extended[c+3], fmt='none') 
        ax.set_xticks([0, 4, 8], ['0.01', '0.09', '0.17'])
        ax.set_ylim(0,7)
        ax.set_xlabel("k")
        ax.set_ylabel("Rate")

def plot_ratedist(ax, rates, re, ri):
    #bins = np.logspace(-2, 2, 50)
    bins = np.linspace(0.01, 35, 70)

    w = np.ones(ri.size) / ri.size
    ax.hist(ri.ravel(),  density=False, weights=w,  histtype='step',  bins=bins, label='L23 inh', color=cr.lcolor['L23_modelI'])
    w = np.ones(re.size) / re.size
    ax.hist(re.ravel(),  density=False, weights=w,  histtype='step',  bins=bins, label='L23 exc', color=cr.lcolor['L23'])

    hist, edges = np.histogram(rates.ravel(), density=False, bins=bins)
    edges = 0.5*(edges[1:] + edges[:-1])

    hist = hist / rates.size

    ax.plot(edges[::2], hist[::2], color=cr.lcolor['L23'], marker='o', ls="--", markersize=cr.ms, zorder=3, label='data exc')

    ax.set_xlabel("Rate")
    ax.set_ylabel('Neuron frac.')

    ax.legend(loc='best', fontsize=9)
    return


def spatial_selectivity_index(ax, rates, ssfe, ssfi):
    bins = np.linspace(0,1,50)

    #Plot results from the model
    w = np.ones(ssfe.size) / ssfe.size
    ax.hist(ssfe, bins=bins, density=False, weights=w, color=cr.lcolor['L23'], histtype='step')

    print("means ", ssfe.mean(), ssfi.mean())
    print("Test result ", ttest_ind(ssfe, ssfi, alternative='greater'))

    w = np.ones(ssfi.size) / ssfi.size
    ax.hist(ssfi, bins=bins, density=False, weights=w, color=cr.lcolor['L23_modelI'], histtype='step')

    #Plot data
    ssf_data = utl.compute_spatial_selectivity_index(rates)
    w = np.ones(ssf_data.size) / ssf_data.size

    hist, edges = np.histogram(ssf_data, density=False, bins=bins)
    edges = 0.5*(edges[1:] + edges[:-1])

    hist = hist / ssf_data.size
    ax.plot(edges[::2], hist[::2], color=cr.lcolor['L23'], marker='o', ls="--", markersize=cr.ms, zorder=3, label='data exc')


    ax.set_xlabel("SSF")
    #ax.set_ylabel("Neuron frac.")


def compute_conn_prob(v1_neurons, v1_connections):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_prepost(v1_neurons, v1_connections)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        meandata[layer] = p["mean"] / np.max(p["mean"])

    return meandata

def compute_prob_dif_layer(units, connections, layer):
    p = np.zeros(15)
    p_err = np.zeros(15)

    ndif = np.zeros(15)

    for kpre in range(8):
        pre_neurons  = fl.filter_neurons(units, layer=layer, tuning='tuned', proofread='ax_clean')
        pre_neurons  = pre_neurons.loc[pre_neurons['pref_ori']==kpre, ['id']].rename(columns = lambda x : f"pre_{x}")
        for kpost in range(8):

            dif = kpre - kpost

            post_neurons = fl.filter_neurons(units, layer='L23', tuning='tuned')
            post_neurons = post_neurons.loc[post_neurons['pref_ori']==kpost, ['id']].rename(columns = lambda x : f"post_{x}")

            pre_neurons['key'] = 1
            post_neurons['key'] = 1
            pairs = pre_neurons.merge(post_neurons, on='key')[['pre_id', 'post_id']]
            selected_connections = fl.synapses_by_id(connections, pre_ids=pre_neurons['pre_id'], post_ids=post_neurons['post_id'], who='both')

            p[dif + 7] += len(selected_connections) / len(pairs)
            p_err[dif + 7] += np.sqrt(p[dif + 7] * (1 - p[dif + 7]) / len(pairs)) 
            ndif[dif + 7] += 1

    p /= ndif
    p_err /= ndif

    maxp = np.max(p) 
    p /= maxp
    p_err /= maxp

    return p, p_err

def compute_prob_dif(units, connections):
    p = {}
    p_err = {}
    
    for layer in ['L23', 'L4']:
        p[layer], p_err[layer] = compute_prob_dif_layer(units, connections, layer)
    
    return p, p_err


def plot_prob_dif(axes, p, perr, pdata, pdataerr):
    x = np.arange(-7, 8)
    for layer in ['L23', 'L4']:

        axes[layer].plot(x, p[layer], color=cr.lcolor[layer], label='Model')

        pmean = pdata[layer]
        pstd = pdataerr[layer] 
        axes[layer].fill_between(x, pmean - pstd, pmean + pstd, color=cr.lcolor[layer], alpha = 0.3)
        axes[layer].plot(x, pmean, color=cr.lcolor[layer], marker='o', ls="--", markersize=cr.ms, zorder=3, label='Data')

        #axes[layer].errorbar(x, p[layer], yerr = perr[layer], color=cr.lcolor[layer], label='Model')
        #axes[layer].errorbar(x, pdata[layer], yerr = pdataerr[layer], color=cr.lcolor[layer], marker='o', ls="--", markersize=cr.ms, zorder=3, label='Data')

        axes[layer].set_xticks([-8, 0, 8], ['-0.17', '0', '0.17'])
        axes['L23'].set_xlabel(r"Δk")

    axes['L23'].set_ylabel(r"Prob. Conn.\n(Normalized)")
    
def plot_probconn(axdata, axmodel, meandata):

    #Plot it!
    axes = {'data': axdata, 'model': axmodel}

    for case in ['data', 'model']:
        im = axes[case].imshow(meandata[case], vmin = 0., vmax = 1.0, extent = [0.01, 0.17, 0.01, 0.17], aspect='auto', origin='lower')

        #Then just adjust axes and put a legend
        axes[case].set_xlabel(r"$\hat k _\text{post}$")
        axes[case].set_xticks([0.01, 0.09, 0.17], ['0.01', '0.09', '0.17'])
        axes[case].set_yticks([0.01, 0.09, 0.17], ['0.01', '0.09', '0.17'])
    
    axes['data'].set_ylabel(r"$\hat k _\text{pre}$")
    return im

#def diff_emergent2target_prefori(ax, pref_ori, target_ori, color):
def diff_emergent2target_prefori(ax, diff_ori, color, label):


    bins = np.arange(-7.5, 8.5)

    hist, edges = np.histogram(diff_ori, bins=bins)
    hist = hist / hist.sum() 

    xvals = 0.5 * (bins[1:] + bins[:-1]) * 0.02125
    av  = np.dot(hist,  xvals**2)
    av2 = np.dot(hist,  xvals**4)


    lines, = ax.step(bins[:-1]+0.5, hist, color=color, label=label)

    ax.set_xlabel(r"$\hat k_\text{targt}- \hat k_\text{emerg}$")
    ax.set_ylabel('Neuron frac.')
    ax.set_xticks([-8, 0, 8], ['-0.17', '0.0', '0.17'])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylim(0., 0.41)
    
    return lines

def make_bar_plot(ax, cvsims, cvdata, title):

    m = cvsims.mean(axis=0)
    s = cvsims.std(axis=0) / np.sqrt(len(cvsims))

    x = np.arange(4)

    ax.bar(x, m, color = cr.reshuf_color, edgecolor='k') 
    ax.errorbar(x, m, yerr = s, color = 'black', marker = 'none', ls='none') 

    ax.axhline(cvdata, ls='--', color = 'gray', lw=1)
    ax.text(x[1], cvdata + 0.005, "Experiment")

    ax.axhline(m[0], ls='--', color = 'black')

    #for i in range(1, 4):
    #    tstat, pval = ttest_ind_from_stats(m[0], s[0], 10, m[i], s[i], 10, alternative='greater')
    #    add_sig_bracket(ax, 0, i, m[0], m[i], yerr1=s[0], yerr2=s[1], p=pval, level=i-1, fs=9)

    tstat, pval = ttest_ind_from_stats(m[2], s[2], 10, m[3], s[3], 10, alternative='two-sided')
    add_sig_bracket(ax, 2, 3, m[2], m[3], yerr1=s[2], yerr2=s[3], p=pval, level=0, fs=9, pad_frac=0.02, h_frac=0.03)

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

def plot_tuning_curve(ax, units, rates, re):

    layer = 'L23'
    neurons_layer = fl.filter_neurons(units, layer=layer, tuning='matched') 
    rates_layer = rates[neurons_layer['id'], :]

    #Mean and its standard error
    tcurve     = np.mean(dutl.shift_multi(rates_layer, neurons_layer['pref_ori']), axis=0) 
    tcurve_err = np.std(dutl.shift_multi(rates_layer, neurons_layer['pref_ori']), axis=0) / np.sqrt(rates_layer.shape[0])

    #tcurve     = plotutils.shift(tcurve, with_symmetric=False)
    #tcurve_err = plotutils.shift(tcurve_err, with_symmetric=False)
    ax.fill_between(np.arange(15), tcurve - tcurve_err, tcurve + tcurve_err, color=cr.lcolor[layer], alpha=0.5, edgecolor=None)
    ax.plot(np.arange(15), tcurve, color=cr.dotcolor[layer], ls="none", marker='o', ms=cr.ms)


    rematrix = np.reshape(re, (-1, 8))
    prefs = np.argmax(rematrix, axis=1)
    tcurve     = np.mean(dutl.shift_multi(rematrix, prefs), axis=0) 
    ax.plot(np.arange(15), tcurve, color=cr.lcolor[layer], label=layer)

    ax.set_xticks([-1, 7, 16], ['-0.17', '0.0', '0.17'])
    ax.set_ylim(0, 10)
    ax.set_xlabel(r"$\hat k - k$")
    ax.set_ylabel("Rate")
    ax.legend(loc='best')

    return 

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 5''')

# Adding and parsing arguments
#parser.add_argument('datafolder', type=str, help='Place where the circular variances are saved')
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname, generate_data=True):

    filename = 'v1300_def_spfreqcosonly'

    nexp = 10 

    # load files
    units, connections, rates, error_rates = loader.load_data(return_error=True)
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)


    #filename = 'best_ale'
    sty.master_format()
    fig, axes = plt.subplot_mosaic(
    """
    AB.CD
    EFZGH
    """,
    figsize=sty.two_col_size(height=8.), layout='constrained', width_ratios=[1., 1., 0.05, 1., 1.]) 

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
            allri = np.empty(0)
            allre = np.empty(0)
            allssfe = np.empty(0)
            allssfi = np.empty(0)
            #probmean = {'L23' : np.zeros((8,8)), 'L4' : np.zeros((8,8))} 
            #proberr = {'L23' : np.zeros((8,8)), 'L4' : np.zeros((8,8))} 
            probmean = {'L23' : np.zeros(15), 'L4' : np.zeros(15)} 
            proberr = {'L23' : np.zeros(15), 'L4' : np.zeros(15)} 

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

                allre = np.concatenate((allre, re.ravel()))
                allri = np.concatenate((allri, ri.ravel()))

                ssftrial = utl.compute_spatial_selectivity_index(re)
                allssfe = np.concatenate((allssfe, ssftrial))
                ssftrial = utl.compute_spatial_selectivity_index(ri)
                allssfi = np.concatenate((allssfi, ssftrial))


                #means = compute_conn_prob(units_sample, connections_sample)
                means, _ = compute_prob_dif(units_sample, connections_sample)
                for layer in ['L23', 'L4']:
                    probmean[layer] += means[layer]
                    proberr[layer] += means[layer]**2

            for layer in ['L23', 'L4']:
                probmean[layer] /= nexp
                proberr[layer] /= nexp
                proberr[layer] -= probmean[layer]**2
                proberr[layer] = np.sqrt(proberr[layer])

            np.save(f"{args.save_destination}/{figname}_{i}_angl_data", diff_ori)
            np.save(f"{args.save_destination}/{figname}_{i}_rateE_data", allre)
            np.save(f"{args.save_destination}/{figname}_{i}_rateI_data", allri)
            np.save(f"{args.save_destination}/{figname}_{i}_ssfE_data", allssfe)
            np.save(f"{args.save_destination}/{figname}_{i}_ssfI_data", allssfi)
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
            allre = np.load(f"{args.save_destination}/{figname}_{i}_rateE_data.npy")
            allri = np.load(f"{args.save_destination}/{figname}_{i}_rateI_data.npy")
            allssfe   = np.load(f"{args.save_destination}/{figname}_{i}_ssfE_data.npy")
            allssfi   = np.load(f"{args.save_destination}/{figname}_{i}_ssfI_data.npy")
            probmean['L23'] = np.load(f"{args.save_destination}/{figname}_{i}_probmeanL23.npy")
            proberr['L23']  = np.load(f"{args.save_destination}/{figname}_{i}_proberroL23.npy")
            probmean['L4']  = np.load(f"{args.save_destination}/{figname}_{i}_probmeanL4.npy")
            proberr['L4']   = np.load(f"{args.save_destination}/{figname}_{i}_proberroL4.npy")

        #First time we need to resize this to the number of Exc neurons in the simulation, which was not known a priori
        if reshuffle_mode == '':
            ssfcomp.resize((allssfe.shape[0], 4))
        
        ssfcomp[:, i] = allssfe

        #diff_emergent2target_prefori(axes['A'], exc_pref_ori, target_ori, c23)    
        handle = diff_emergent2target_prefori(axes['G'], diff_ori, c23, label)    
        legend_handles.append(handle)


        if reshuffle_mode == '':
            plot_tuning_curve(axes['B'], units, rates, allre)
            spatial_selectivity_index(axes['D'], rates, allssfe, allssfi)

            dataconprob = compute_conn_prob(matched_neurons, matched_connections)
            #im = plot_probconn(axes['E'], axes['F'], {'data': dataconprob['L23'], 'model' : probmean['L23']})
            #cbar = fig.colorbar(im, cax=axes['Z'])

            probmeandata, proberrdata = compute_prob_dif(units, connections)
            plot_prob_dif({'L23' : axes['E'], 'L4' : axes['F']}, probmean, proberr, probmeandata, proberrdata)


            plot_ratedist(axes['C'], rates, allre, allri)


        #spatial_selectivity_index(axes['C'], allssf, c23)

        #cvcomp = np.vstack((cvcomp, allcircv))

        #TODO will need to be plotted somehow...
        #conn_prob_osi(axes['D'], axes['E'], probmean, proberr, c23, c23)

    example_tuning_curve(axes['A'], units, rates, error_rates)
    




    units_e = fl.filter_neurons(units, layer='L23', tuning='matched', cell_type='exc')
    ssf_data = utl.compute_spatial_selectivity_index(rates[units_e['id']])
    aver_ssf = ssf_data.mean()

    #axes['H'].set_axis_off()
    make_bar_plot(axes['H'], ssfcomp, aver_ssf, "")
    #axes['A'].legend(handles=legend_handles, loc=(0.1, 0.55))


    #axes2label = [axes[key] for key in 'ABCDEL']
    #label_pos  = [-0.25, 1.05] * 6 
    #sty.label_axes(axes2label, label_pos)
    

    fig.savefig(f"{args.save_destination}/{figname}.pdf",  bbox_inches="tight")

plot_figure("fig6cosonly",  generate_data=True)