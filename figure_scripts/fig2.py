import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.stats import sem
from scipy.stats import wilcoxon, mannwhitneyu
import sys
sys.path.append(".")
from ccmodels.analysis.utils import tuning_encoder
from ccmodels.plotting.utils import  prob_conectivity_tuned_untuned, prob_conn_diffori, strength_tuned_untuned, cumulative_probconn 
import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr





def plot_dist(ax, data, layer):

    #Filter the data
    neurons_layer = data[data['cortex_layer'] == layer]

    #Create bins and compute their centers, which is useful for plotting
    bins = np.linspace(0, 1, 20)
    bins_centered = 0.5*(bins[1:] + bins[:-1])

    #Select only tuned neurons
    tuned_neurons = neurons_layer[(neurons_layer['model_type'] == 'orientation') | (neurons_layer['model_type'] == 'direction')]['osi']

    #Histogram them, normalizing to count (not by density) 
    hist, _ = np.histogram(tuned_neurons, bins)
    hist = hist/np.sum(hist)
    ax.step(bins_centered, hist, color = cr.lcolor[layer])


def fraction_tuned(ax, data):
    barw = 0.1
    ybars = [0, barw] 
    offset = 0.05 #To display text

    #Create a Pandas Series which contains the number of tuned neurons in a layer
    #The value is accesed by the key of the pandas dataframe, e.g. n_tuned["L2/3"]
    tuned_neurons = data[(data['model_type'] == 'orientation') | (data['model_type'] == 'direction')]
    n_tuned = tuned_neurons.groupby("cortex_layer").size()
    total_neurons = data.groupby("cortex_layer").size() 

    layers = ["L2/3", "L4"]

    #Plot those fractions
    for i,layer in enumerate(["L2/3", "L4"]):
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
    

def plot_matrix_tuneuntune(ax, averages, addticks=False, title=""):
    ax.imshow(averages, interpolation="none")

    for (j,i),value in np.ndenumerate(averages):
        ax.text(i, j, f"{value:.2f}", ha="center",va="center", color="white")
    
    ax.set_yticks([0,1], labels=["L2/3T", "L2/3U"])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.set_title(title)

    if addticks:
        ax.set_xticks([0,1,2,3], labels=["L2/3T", "L2/3U", "L4T", "L4U"])
    else:
        ax.set_xticks([])
    


def conn_prob_osi(ax, data):

    #Get the data to be plotted 
    conprob = {}
    conprob["L2/3"], conprob["L4"]= prob_conn_diffori(data)

    #Plot it!
    for layer in ["L2/3", "L4"]:
        p = conprob[layer]
        c = cr.lcolor[layer]

        ax.fill_between(p['directions'],p ['mean']-p['std'], p['mean']+p['std'], color = c, alpha = 0.2)
        ax.plot(p['directions'], p['mean'], color = c, label = layer)
        ax.scatter(p['directions'], p['mean'], color = 'black', s=5, zorder = 3)


    #Then just adjust axes and put a legend
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel('∆ori')
    ax.set_ylabel("Conn. Probability")
    ax.legend(loc = 'upper right')

    # Significance band
    ax.annotate('***', xy=(0.8, 0.043), xytext=(0.8, 0.035), xycoords='data', ha='center', va='bottom', 
                arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1', lw=2.0, color='k'))


#TODO we have to improve the return format of this function to allow colors, etc to be better...
def plot_cumulative(ax, data):
    cumul_dists = cumulative_probconn(data, [0, 1.570796, 0, 1.570796])
    labels = ["L2/3, θ=0", "θ=π", "L4, θ=0", "θ=π"]

    for i,cd in enumerate(cumul_dists):
        color = cr.angles[i]
        label = labels[i]
        ax.step(cd[1], cd[0]/np.sum(cd[0][-1]), color = color, label = label)

    ax.set_xlabel("Conn. Strength")
    ax.set_ylabel("Cumulative")
    ax.set_xscale("log")
    ax.legend(loc=(0.05, 0.65))


# ----------------------------------------------------------------------------------------

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

#fig, axes = plt.subplot_mosaic(
#    """
#    ABC
#    DBC
#    EFG
#    """, 
#    figsize=sty.two_col_size(ratio=1.5), layout="constrained", gridspec_kw={"height_ratios":[0.15,0.85,1], })

#Read and process necessary data
v1_neurons = pd.read_pickle('../con-con-models/data/v1l234_neurons.pkl')
v1_connections = pd.read_pickle('../con-con-models/data/v1l234_connections.pkl')

#Encoding numerically if input and output is tuned or untuned
v1_connections = tuning_encoder(v1_connections,'pre_type', 'post_type', 'not_selective')

# --- First panel

#Leave space for the diagram
axes["left"][0].axis("off")
axes["center"][0].axis("off")


#Plot the data for both layer in the same axis. Then format it. 
fraction_tuned(axes["left"][1], v1_neurons)


#Plot the data for both layer in the same axis. Then format it. 
ax = axes["left"][2]
plot_dist(ax, v1_neurons, "L2/3")
plot_dist(ax, v1_neurons, "L4")

ax.set_ylabel('Fraction')
ax.set_xlabel('OSI')
ax.set_ylim(bottom = 0)

# ----

conn_probability_matrix = prob_conectivity_tuned_untuned(v1_connections)
strength_matrix = strength_tuned_untuned(v1_connections)
plot_matrix_tuneuntune(axes["center"][1], conn_probability_matrix, title="Conn. Probability", addticks=True)
plot_matrix_tuneuntune(axes["center"][2], strength_matrix, title="Conn. Strength")

# --------- 

conn_prob_osi(axes["right"][0], v1_connections)

# --------

plot_cumulative(axes["right"][1], v1_connections)





fig.savefig(args.save_destination+"fig2.pdf", bbox_inches="tight")