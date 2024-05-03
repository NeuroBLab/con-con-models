import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
sys.path.append(".")

#from ccmodels.analysis.utils import tuning_encoder
import ccmodels.dataanalysis.utils as utl 

import ccmodels.dataanalysis.statistics_extraction as ste 

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr
import ccmodels.plotting.utils as plotutils

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl


# ======================================================
# --------------- FUNCTIONS TO DRAW --------------------
# Each function draws a panel/subpanel of the figure. 
# It takes the axis to draw and draws there. In this way
# our figure code is "modular".
# ======================================================




def plot_dist(ax, v1_neurons, layer):

    #Select tuned neurons in the desired layer
    tuned_neurons = fl.filter_neurons(v1_neurons, tuning="tuned", layer=layer)

    #Create bins and compute their centers, which is useful for plotting
    bins = np.linspace(0, 1, 20)
    bins_centered = 0.5*(bins[1:] + bins[:-1])

    #Filter the data
    #tuned_neurons = utl.get_tuned_neurons(neurons_layer)

    #Histogram them, normalizing to count (not by density) 
    hist, _ = np.histogram(tuned_neurons["osi"], bins)
    hist = hist/np.sum(hist)
    ax.step(bins_centered, hist, color = cr.lcolor[layer])


def fraction_tuned(ax, data):
    barw = 0.1
    ybars = [0, barw] 
    offset = 0.05 #To display text

    #Create a Pandas Series which contains the number of tuned neurons in a layer
    #The value is accesed by the key of the pandas dataframe, e.g. n_tuned["L2/3"]
    #tuned_neurons = utl.get_tuned_neurons(data)
    tuned_neurons = fl.filter_neurons(data, tuning="tuned") 
    n_tuned = tuned_neurons.groupby("layer").size()
    total_neurons = data.groupby("layer").size() 

    #TODO check
    #matched_neurons = fl.filter_neurons(data, tuning="matched") 
    #total_neurons = matched_neurons.groupby("layer").size() 

    layers = ["L23", "L4"]

    #Plot those fractions
    for i,layer in enumerate(["L23", "L4"]):
        perc_tuned = n_tuned[layer]/total_neurons[layer]
        ax.barh(ybars[i], perc_tuned, color=cr.lcolor[layer], height=barw)
        ax.text(perc_tuned + offset, ybars[i], f"{round(100*perc_tuned)}%", va="center", ha="left")
    
    #Configure the axis in a nice way  
    #No spine below, but mark the 100% with a vertical line
    ax.set_yticks(ybars, layers) 
    ax.set_xticks([0, 1], labels=["0", "100%"])
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(1, color="black", lw=3)
    ax.set_xlabel("% of tuned neurons", fontsize=8)
    

def plot_matrix_tuneuntune(ax, averages_dict, addticks=False, title=""):

    #Store the results as a numpy array for plotting
    averages = np.empty((2,4))
    averages[:, 0] = [averages_dict[f"l23t_l23{x}"] for x in "tu"]
    averages[:, 1] = [averages_dict[f"l23u_l23{x}"] for x in "tu"]
    averages[:, 2] = [averages_dict[f"l4t_l23{x}"]  for x in "tu"]
    averages[:, 3] = [averages_dict[f"l4u_l23{x}"]  for x in "tu"]

    #Plot it!
    ax.imshow(averages, interpolation="none")

    #Plot the text inside of the matrix
    for (j,i),value in np.ndenumerate(averages):
        ax.text(i, j, f"{value:.2f}", ha="center",va="center", color="white")
    
    #Beautiful ticks and so on...
    ax.set_yticks([0,1], labels=["L2/3T", "L2/3U"])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.set_title(title)

    if addticks:
        ax.set_xticks([0,1,2,3], labels=["L2/3T", "L2/3U", "L4T", "L4U"])
    else:
        ax.set_xticks([])
    


def conn_prob_osi(ax, v1_neurons, v1_connections, half=True):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"]= ste.prob_conn_diffori(v1_neurons, v1_connections, half=half)

    #Plot it!
    angles = plotutils.get_angles(kind="diff", half=orientation_only)

    for layer in ["L23", "L4"]:
        p = conprob[layer]
        c = cr.lcolor[layer]

        low_band  = p['mean'] - p['std']
        high_band = p['mean'] + p['std']
        meandata = p['mean']

        low_band  = plotutils.add_symmetric_angle(low_band.values)
        high_band = plotutils.add_symmetric_angle(high_band.values)
        meandata  = plotutils.add_symmetric_angle(meandata.values)

        ax.fill_between(angles, low_band, high_band, color = c, alpha = 0.2)

        ax.plot(angles, meandata, color = c, label = layer)
        ax.scatter(angles, meandata, color = 'black', s=5, zorder = 3)

    ax.axvline(0, color="gray", ls=":")

    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel('∆ori')
    ax.set_ylabel("Conn. Probability")

    plotutils.get_xticks(ax, max=2*np.pi, half=True)

    ax.legend(loc = 'upper right')


def plot_cumulative(ax, v1_neurons, v1_connections):
    cumul_dists = ste.cumulative_probconn(v1_neurons, v1_connections, [0, 4, 0, 4])
    labels = ["L2/3, θ=0", "θ=π", "L4, θ=0", "θ=π"]

    for i,cd in enumerate(cumul_dists):
        color = cr.angles[i]
        label = labels[i]
        ax.step(cd[1], cd[0]/np.sum(cd[0][-1]), color = color, label = label)

    ax.set_xlabel("Conn. Strength")
    ax.set_ylabel("Cumulative")
    ax.set_xscale("log")
    ax.legend(loc=(0.05, 0.65))




# ======================================================
# --------------- FIGURE STRUCTURE ---------------------
# THis is the code that loads the data, structures the 
# location of the panels, and then call the analysis 
# functions to fill in the panels, via the functions above.
# ======================================================



#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

#Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()

sty.master_format()

fig = plt.figure(layout="constrained", figsize=sty.two_col_size(ratio=1.5))
subfigs = fig.subfigures(1, 3)

axes = {}

axes["left"] = subfigs[0].subplots(nrows=3, ncols=1, height_ratios=[1, 0.15, 0.85])
axes["center"] = subfigs[1].subplots(nrows=3, ncols=1, height_ratios=[1, 0.5, 0.5])
axes["right"] = subfigs[2].subplots(nrows=2, ncols=1)

#Load the data
orientation_only = True 
v1_neurons, v1_connections, rates = loader.load_data(half_angle=orientation_only)

#Se we can easily filter synapses by the layer in which the presynaptic neuron lives directly
#without having to call .isin(...) all the time.
#Adds two extra columns to v1_connections
v1_connections = utl.add_layerinfo_to_connections(v1_neurons, v1_connections, who="pre") 

#For many things in this figure we need only the functionally matched neurons, the others are not useful
matched_neurons = fl.filter_neurons(v1_neurons, tuning="matched")
matched_connections = fl.synapses_by_id(matched_neurons["id"], v1_connections, who="both")

# --- First panel

#Leave space for the diagram
axes["left"][0].axis("off")
axes["center"][0].axis("off")


#Plot the data for both layer in the same axis. Then format it. 
fraction_tuned(axes["left"][1], matched_neurons)

#Plot the data for both layer in the same axis. Then format it. 
ax = axes["left"][2]
plot_dist(ax, matched_neurons, "L23")
plot_dist(ax, matched_neurons, "L4")

#Nice labels
ax.set_ylabel('Fraction')
ax.set_xlabel('OSI')
ax.set_ylim(bottom = 0)

# ----

#Get the dictinoarie sof both probability and strength
conn_probability_dict = ste.prob_conectivity_tuned_untuned(matched_neurons, matched_connections)
strength_dict = ste.strength_tuned_untuned(matched_neurons, matched_connections)

#Make the plots
plot_matrix_tuneuntune(axes["center"][1], conn_probability_dict, title="Conn. Probability", addticks=True)
plot_matrix_tuneuntune(axes["center"][2], strength_dict, title="Conn. Strength")

# --------- 

#Probability as a function of the dtheta
conn_prob_osi(axes["right"][0], matched_neurons, matched_connections, half=orientation_only)

# --------

plot_cumulative(axes["right"][1], matched_neurons, matched_connections)





fig.savefig(args.save_destination+"fig2.pdf", bbox_inches="tight")