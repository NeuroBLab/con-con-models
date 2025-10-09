import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

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

import ccmodels.utils.angleutils as au


#Get a simple tuning curve from the neuron with the selected id.  
def example_tuning_curve(ax, rates, rates_err, id):
    rangle = plotutils.shift(rates[id, :])
    rangle_err = plotutils.shift(rates_err[id, :])

    ax.plot(np.arange(9), rangle,  lw=1, color=cr.lcolor['L23'])
    ax.plot(np.arange(9), rangle,  lw=1, color=cr.dotcolor['L23'], ls='none', marker='o', ms=cr.ms)
    ax.errorbar(np.arange(9), rangle,  yerr = rangle_err , color=cr.dotcolor['L23'], fmt='none')

    ax.set_xticks([0, 4, 8], [0, 'π/2', 'π'])
    ax.set_xlabel("θ")
    ax.set_ylabel("Rate")

#Plot an example current to the postsynaptic neuron with the selected id
def example_current(ax, v1_neurons, connections, vij, rates, rates_err, id):

    currents = {}
    currents_err = {}
    
    #Get the presynaptic filtering and compute the current 
    pre_ids = fl.filter_neurons(v1_neurons, layer='L23', tuning='matched', proofread='minimum')
    currents['L23'] = curr.get_currents_subset(v1_neurons, vij, rates, post_ids=[id], pre_ids=pre_ids['id'], shift=False)[0]
    currents_err['L23'] = curr.get_currents_subset(v1_neurons, vij, rates_err, post_ids=[id], pre_ids=pre_ids['id'], shift=False)[0]

    #Then repeat for the other layer
    pre_ids = fl.filter_neurons(v1_neurons, layer='L4', tuning='matched', proofread='minimum')
    currents['L4'] = curr.get_currents_subset(v1_neurons, vij, rates, post_ids=[id], pre_ids=pre_ids['id'], shift=False)[0]
    currents_err['L4'] = curr.get_currents_subset(v1_neurons, vij, rates_err, post_ids=[id], pre_ids=pre_ids['id'], shift=False)[0]

    #And the total current
    pre_ids = fl.filter_neurons(v1_neurons, tuning='matched', proofread='minimum')
    currents['Total'] = curr.get_currents_subset(v1_neurons, vij, rates, post_ids=[id], pre_ids=pre_ids['id'], shift=False)[0]
    currents_err['Total'] = curr.get_currents_subset(v1_neurons, vij, rates_err, post_ids=[id], pre_ids=pre_ids['id'], shift=False)[0]
    maxcurr = np.max(currents['Total'])


    #For each one of the computed things, just plot it
    for layer in currents:
        currents[layer]     = currents[layer] / maxcurr
        currents_err[layer] = currents_err[layer] / maxcurr
        shiftcur = plotutils.shift(currents[layer] )
        shiftcur_err = plotutils.shift(currents_err[layer] )
        ax.errorbar(np.arange(9), shiftcur, yerr=shiftcur_err, color=cr.lcolor[layer])


    ax.set_xticks([0, 4, 8], [0, 'π/2', 'π'])
    ax.set_xlabel("θ")
    ax.set_ylabel("Synaptic current μ(θ)")

def show_image(ax, path2im):
    im = Image.open("images/" + path2im)
    ax.imshow(im)

def plot_currents(axes, units, rates, vij):

    #Get the postsynaptic tuned neurons
    post = fl.filter_neurons(units, layer='L23', tuning='tuned')
    
    #Get the presynaptic neurons in each one of the layers + the total one
    pre = {}
    pre["Total"]  = fl.filter_neurons(units, proofread='minimum')
    for layer in ["L23", "L4"]:
        pre[layer]  = fl.filter_neurons(units, layer=layer, proofread='minimum')
    
    #To store the results...
    avgcurrent = {}
    stdcurrent = {}

    #We will need the maximum total current to normalize
    current = curr.get_currents_subset(units, vij, rates, post_ids=post['id'], pre_ids=pre["Total"]['id'], shift=True)
    avgcurrent['Total'] = np.mean(current, axis=0)
    maxcur = np.max(avgcurrent['Total'])

    #Just to plot
    x = np.arange(9)

    legendlabel = ["L2/3", 'L4', "Total"]

    #Now for each one of the layers,
    for i,layer in enumerate(['L23', 'L4', "Total"]):
        #Get the presynaptic people for this case
        pre_layer = pre[layer]

        #Compute all the currents to postsynaptic neurons 
        current = curr.get_currents_subset(units, vij, rates, post_ids=post['id'], pre_ids=pre_layer['id'], shift=True)

        #Get the mean and its error
        n = np.sqrt(current.shape[0])
        avgcurrent[layer] = np.mean(current, axis=0)
        stdcurrent[layer] = np.std(current, axis=0) / n

        #Shift for plotting, normalize to maximum total current
        avgcurrent[layer] = plotutils.shift(avgcurrent[layer] / maxcur)
        stdcurrent[layer] = plotutils.shift(stdcurrent[layer] / maxcur)

        #Plot 
        axes[0].fill_between(x, avgcurrent[layer] - stdcurrent[layer], avgcurrent[layer] + stdcurrent[layer], color=cr.lcolor[layer], alpha=0.5, edgecolor=None)
        axes[0].plot(x, avgcurrent[layer], color=cr.lcolor[layer], label=legendlabel[i])     
        axes[0].plot(x, avgcurrent[layer], color=cr.dotcolor[layer], ms=cr.ms, ls='none', marker='o')

        #Get the normalization for each curve
        maxcurlayer = np.max(avgcurrent[layer])
        avgcurrent[layer] = avgcurrent[layer] / maxcurlayer 
        stdcurrent[layer] = stdcurrent[layer] / maxcurlayer 

        #Replot the normalized thing
        axes[1].fill_between(x, avgcurrent[layer] - stdcurrent[layer], avgcurrent[layer] + stdcurrent[layer],color=cr.lcolor[layer], alpha=0.5, edgecolor=None) 
        axes[1].plot(x, avgcurrent[layer], color=cr.lcolor[layer])     
        axes[1].plot(x, avgcurrent[layer], color=cr.dotcolor[layer], ms=cr.ms, ls='none', marker='o')

    for ax in axes:
        ax.set_xticks([0, 4, 8], ['-π/2', 0, 'π/2'])
        ax.set_xlabel("Δθ")

    axes[0].set_ylabel("μ(Δθ)")
    axes[1].set_ylabel("μ(Δθ)/μ(0)")

def prediction_shuffling_control(ax, units, connections, rates, vij, nreps = 1000):

    rates = dutl.get_untuned_rate(units, rates) 

    #Initialize 
    pre = {}
    post = {}
    in_degree = {}
    weights = {}

    #Prediction and observed pref oris
    pref_oris_pred = {}
    pref_ori_data = {}
    #Difference between them and fraction correct
    delta_target_pred_data = {}
    fraction_correct = {}
    abs_error = {}

    #Get connections from proofread presynaptic neurons to tuned L23 ones
    synapses = fl.filter_connections_prepost(units, connections, layer = [None, "L23"], tuning=[None, 'tuned'], proofread=["minimum", None])
    pre["Total"] = synapses["pre_id"].unique()
    post["Total"] = synapses["post_id"].unique()
    in_degree["Total"] = synapses['post_id'].value_counts().sort_index().values
    weights["Total"] = synapses["syn_volume"]

    #Unique does not sort the indices, but we will them to be sorted 
    pre["Total"].sort()
    post["Total"].sort()

    #Repeat the very same thing but using presynaptic layers L23 and L4
    for layer in ["L23", "L4"]:
        synapses = fl.filter_connections_prepost(units, connections, layer = [layer, "L23"], tuning=[None, 'tuned'], proofread=["minimum", None])
        pre[layer]   = synapses['pre_id'].unique()
        post[layer]  = synapses['post_id'].unique()
        in_degree[layer] = synapses['post_id'].value_counts().sort_index().values
        weights[layer] = synapses["syn_volume"]

        pre[layer].sort()
        post[layer].sort()



    #Compute the difference in target vs predicted for our data
    for layer in ["L23", "L4", "Total"]:
        #Compute all the currents to postsynaptic neurons 
        currents_data = curr.get_currents_subset(units, vij, rates, post_ids=post[layer], pre_ids=pre[layer], shift=False)

        #Get their predicted pref ori and then compare with the actual postsynp neurons'  pref ori 
        pref_oris_pred[layer] = np.argmax(currents_data, axis=1)
        pref_ori_data[layer] = units.loc[units['id'].isin(post[layer]), 'pref_ori'].values
        delta_target_pred_data[layer] = au.angle_dist(pref_oris_pred[layer], pref_ori_data[layer])

        #Initialize for bootstrap
        fraction_correct[layer] = np.empty(nreps)
        abs_error[layer] = np.empty(nreps)

    #Save that last one result from the loop: the prediction of pref_oris using the currents for all presynaptic neurons
    #pref_oris_pred_total = pref_oris_pred.copy()
    fraction_shuffled = np.empty(nreps) 
    abs_error_shuffled = np.empty(nreps)

    #Bootstrap for n repetitions 
    for i in range(nreps):

        #For each layer, get a bootstrap of our data currents to estimate mean and error
        for layer in ["Total", "L23", "L4"]:
            #Bootstrap (shortened to btrp) our real data
            bootstraped_diff_angle = np.random.choice(delta_target_pred_data[layer], replace=True, size=len(delta_target_pred_data[layer]))

            #Then, get the fractoin of good predictions in this bootstrapped sample and the mean difference
            fraction_correct[layer][i] = (bootstraped_diff_angle == 0).sum() / len(bootstraped_diff_angle) 
            abs_error[layer][i] = bootstraped_diff_angle.mean() * np.pi / 8



        layer = "Total"
        shuffled_post_oris = np.empty(len(post[layer]))
        for pix in range(len(post[layer])):
            k = in_degree["Total"][pix]
            w = np.random.choice(weights[layer], replace=True, size=k)
            pre_random = np.random.choice(len(pre[layer]), replace=True, size=k)
            current = np.dot(w, rates[pre_random, :]) 
            shuffled_post_oris[pix] = np.argmax(current)
        
        diff_angles = au.angle_dist(shuffled_post_oris, pref_ori_data[layer])
        fraction_shuffled[i] = (diff_angles == 0).sum() / len(diff_angles) 
        abs_error_shuffled[i] = diff_angles.mean() * np.pi / 8


    #Position of the random level
    linepos = abs_error_shuffled.mean() 
    ax.axhline(linepos, color="black")

    xoffset = -0.1
    yoffset = 0.01

    barpos = np.arange(3)

    #Compute all currents to the postsynaptic neurons
    for i,layer in enumerate(['L23', 'L4', "Total"]):

        #delta = fraction_correct[layer].mean() 
        #delta_err = fraction_correct[layer].std() 

        delta = abs_error[layer].mean() 
        delta_err = abs_error[layer].std() 

        #For each bootstrap test, is the null hypothesis as large as our observation?
        #The fraction of times this happened is the pvalue
        p = np.mean(abs_error_shuffled <= abs_error[layer])

        print(abs_error_shuffled.mean())

        if p < 0.001:
            sign = '***'
        elif p < 0.01:
            sign = '**'
        elif p < 0.05:
            sign = '*'
        else:
            sign = 'n.s.'

        ax.text(barpos[i] + xoffset + 0.2 * xoffset * len(sign), linepos + yoffset, f"{sign}")

        ax.bar(barpos[i], delta, yerr=delta_err, color=cr.lcolor[layer], edgecolor='k')


    ax.set_yticks([0, np.pi/8, np.pi/4], ['0', 'π/8', 'π/4'])
    #ax.set_ylim(np.pi/6, np.pi/3.75)
    ax.set_ylim(0, np.pi/3.25)
    ax.set_xticks(barpos, ['L2/3', 'L4', 'Total'])

    return


#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

def plot_figure(figname):
    # load files
    units, connections, rates, rates_err = loader.load_data(return_error=True)
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    matched_neurons = fl.filter_neurons(units, tuning="matched")
    matched_connections = fl.synapses_by_id(connections, pre_ids=matched_neurons["id"], post_ids=matched_neurons["id"], who="both")

    vij = loader.get_adjacency_matrix(matched_neurons, matched_connections)



    sty.master_format()
    fig = plt.figure(figsize=sty.two_col_size(ratio=1.9), layout='constrained')
    #ghostax = fig.add_axes([0,0,1,1])
    
    #ghostax.axis('off')

    axes = fig.subplot_mosaic(
        #[['T', 'T',  'T', 'T',  'T'],
        # ['X', 'A1', '.', 'B1', 'C'],
        # ['X', 'A2', '.', 'B2', 'L']],
        # width_ratios=[1,1,0.3,1,1], height_ratios=[1, 1], empty_sentinel='.'
        [['T', 'T', 'T',  'T'],
         ['X', 'A1','B1', 'C'],
         ['X', 'A2','B2', 'L']],
         height_ratios=[0.5, 1, 1],
          width_ratios=[1,1,1,1]
    )

    #USe the upper space for titles
    axes['T'].set_axis_off()
    axes['T'].text(0.25, 1., 'Example neuron',    weight='bold', ha='center')
    axes['T'].text(0.75, 1., 'Post-syn. average', weight='bold', ha='center')
    axes['T'].set_in_layout(False)

    #units = fl.filter_neurons(units, layer='L23', tuning='tuned', proofread='dn_clean')
    selected_unit = 19 #1119
    id = units['id'].values[selected_unit] 

    axes['X'].set_axis_off()
    axes['X'].text(0.1, 0.9, f"pt_root_id:\n{units['pt_root_id'].values[selected_unit]}", fontsize=10)


    example_tuning_curve(axes['A1'], rates, rates_err, id)
    example_current(axes['A2'], matched_neurons, matched_connections, vij, rates, rates_err, id)
    plot_currents([axes['B1'], axes['B2']], matched_neurons, rates, vij)


    #plot_sampling_current(axes['B1'], axes['B2'], matched_neurons, matched_connections, rates)
    prediction_shuffling_control(axes['C'], matched_neurons, matched_connections, rates, vij)

    axes['L'].set_axis_off()
    handles, labels = axes['B1'].get_legend_handles_labels()
    axes['L'].legend(handles, labels, loc=(0., 0.5), handlelength=1.2)

    axes2label = [axes[k] for k in ['A1', 'B1', 'C']]
    label_pos  = [[0.1, 0.95]] * 3 
    sty.label_axes(axes2label, label_pos)
    fig.savefig(f"{args.save_destination}/{figname}",  bbox_inches="tight")


plot_figure("fig2.pdf")
