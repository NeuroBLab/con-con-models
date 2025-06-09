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


# ------------------ Plotting functinos -------------------------------------# 

def make_plot(ax, cvsims, cvdata, title):

    m = cvsims.mean(axis=0)
    s = cvsims.std(axis=0) / np.sqrt(len(cvsims))

    x = np.arange(4)

    ax.bar(x, m, color = cr.reshuf_color) 
    ax.errorbar(x, m, yerr = s, color = 'black', marker = 'none', ls='none') 

    ax.axhline(cvdata, ls='--', color = 'black')
    ax.text(x[1], cvdata + 0.005, "CircVar experiment")

    ax.set_xticks(x, ['Original', 'All Reshf.', 'L23 Reshf.', 'L4 Reshf.'])
    ax.set_ylim(0, 0.4)

    ax.set_title(title)
    return


# ---------------------------- Figure code --------------------------------

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('datafolder', type=str, help='Folder where the files are located')
parser.add_argument('save_destination', type=str, help='PDF output')
args = parser.parse_args()

def plot_figure(figname):

    cvnormal = np.loadtxt(f"data/model/simulations/{args.datafolder}/cvnormal.txt")
    cvtuned  = np.loadtxt(f"data/model/simulations/{args.datafolder}/cvtuned.txt")

    units, connections, rates = loader.load_data()
    connections = fl.remove_autapses(connections)
    connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

    #Get all the data and store in a dict
    summary_data = {}
    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
    _, cvdexp = utl.compute_circular_variance(rates, orionly=True)
    cvdexp = cvdexp.mean()

    #Do the figure
    sty.master_format()
    fig, axes = plt.subplots(figsize=sty.two_col_size(height=9.5), ncols=2, layout="constrained")

    make_plot(axes[0], cvnormal, cvdexp, "Untuned inh")
    make_plot(axes[1], cvtuned, cvdexp, "Tuned inh")

    axes[0].set_ylabel("Aver. Circ. Var.")

    #Label the axes
    axes2label = [axes[k] for k in range(2)] 
    #label_pos  = [0.8, 0.9] * 5 
    label_pos  = [-0.25, 1.05] * 6 
    sty.label_axes(axes2label, label_pos)

    fig.savefig(f"{args.save_destination}/{figname}.pdf",  bbox_inches="tight")


plot_figure("cvcompbar")