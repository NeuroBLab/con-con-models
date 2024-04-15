import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys
sys.path.append(".")

import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.utils as utl

import ccmodels.plotting.styles as sty 
import ccmodels.plotting.color_reference as cr

#Defining Parser
parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')

# Adding and parsing arguments
parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
args = parser.parse_args()


def show_image(ax, path2im):
    im = Image.open("images/" + path2im)

    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def input_statistics(ax):
      #Get the data we want to plot
      nonproof_inputs_counts, proof_inputs_counts, nonproof_outputs_counts, proof_outputs_counts = utl.get_number_connections()

      #Get the medians of the data
      proof_medians = [np.median(values) for values in [proof_inputs_counts, proof_outputs_counts]]
      nonpr_medians = [np.median(values) for values in [nonproof_inputs_counts, nonproof_outputs_counts]]

      #Position x of the bars
      barwidth = 1
      xpos = np.array([1, 4])
      xpos_proof = [x - barwidth/2 for x in xpos]
      xpos_nonpr = [x + barwidth/2 for x in xpos]
      
      #Plot the bars 
      ax.bar(xpos_proof, proof_medians, color="green", label="Proofread")
      ax.bar(xpos_nonpr, nonpr_medians, color="red", label="Non proofread")

      #Legend, title and labels
      ax.set_xticks(xpos, labels=["Input", "Output"])
      ax.set_title("Median connectivity")

def fractional_statistics(ax):
      #Get the data we want to plot
      boots_propl_proof, boots_propl_noproof = utl.get_propotion_connections()

      #Get the medians of the data
      proof_means = [np.median(boots_propl_proof[layer]) for layer in ["L2/3", "L4"]]
      nonpr_means = [np.median(boots_propl_noproof[layer]) for layer in ["L2/3", "L4"]]
      proof_std   = [np.std(boots_propl_proof[layer]) for layer in ["L2/3", "L4"]]
      nonpr_std   = [np.std(boots_propl_noproof[layer]) for layer in ["L2/3", "L4"]]

      #Position x of the bars
      barwidth = 1
      xpos = np.array([1, 4])
      xpos_proof = [x - barwidth/2 for x in xpos]
      xpos_nonpr = [x + barwidth/2 for x in xpos]
      
      #Plot the bars 
      ax.bar(xpos_proof, proof_means, color="green", yerr=proof_std, label="Proofread")
      ax.bar(xpos_nonpr, nonpr_means, color="red", yerr=nonpr_std, label="Non proofread")

      #Legend, title and labels
      ax.legend(loc=(0.4, 0.6))
      ax.set_xticks(xpos, labels=["L2/3", "L4"])
      ax.set_title("Median input proportion")


sty.master_format()
fig, axes = plt.subplot_mosaic(
    """
    ABF
    CDD
    CDD
    """,
    figsize=sty.two_col_size(ratio=2), layout="constrained",
    gridspec_kw={"width_ratios":[1, 0.7, 0.7], "height_ratios":[1.2,1,1]}
)

show_image(axes["A"], "network_schema.png")
show_image(axes["C"], "3d_reconstruction.png")
show_image(axes["D"], "fig1_plotE.png")

input_statistics(axes["B"])
fractional_statistics(axes["F"])

#Separation between axes
fig.get_layout_engine().set(wspace=1/72, w_pad=0)

fig.savefig(args.save_destination+"fig1.pdf",  bbox_inches="tight")



