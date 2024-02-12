import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys
sys.path.append(".")
from ccmodels.plotting.utils import get_number_connections, get_propotion_connections 
import ccmodels.plotting.styles as sty 

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
      nonproof_inputs_counts, proof_inputs_counts, nonproof_outputs_counts, proof_outputs_counts = get_number_connections()

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
      ax.legend(loc=(0.2, 0.6))
      ax.set_xticks(xpos, labels=["Input", "Output"])
      ax.set_title("Median connectivity")

def fractional_statistics(ax):
      #Get the data we want to plot
      boots_propl_proof, boots_propl_noproof = get_propotion_connections()

      #Get the medians of the data
      proof_means = [np.mean(boots_propl_proof[layer]) for layer in ["L2/3", "L4"]]
      nonpr_means = [np.mean(boots_propl_noproof[layer]) for layer in ["L2/3", "L4"]]
      proof_std   = [np.std(boots_propl_proof[layer]) for layer in ["L2/3", "L4"]]
      nonpr_std   = [np.std(boots_propl_noproof[layer]) for layer in ["L2/3", "L4"]]

      #Position x of the bars
      barwidth = 1
      xpos = np.array([1, 4])
      xpos_proof = [x - barwidth/2 for x in xpos]
      xpos_nonpr = [x + barwidth/2 for x in xpos]
      
      #Plot the bars 
      ax.bar(xpos_proof, proof_means, color="green", yerr=proof_std)
      ax.bar(xpos_nonpr, nonpr_means, color="red", yerr=nonpr_std)

      #Legend, title and labels
      ax.legend(loc=(0.2, 0.6))
      ax.set_xticks(xpos, labels=["L2/3", "L4"])
      ax.set_title("Mean input proportion")


sty.master_format()
fig, axes = plt.subplot_mosaic(
    """
    AB
    CD
    CE
    """,
    figsize=sty.two_col_size(ratio=2), gridspec_kw={"width_ratios":[0.8, 1]}, layout="constrained"
)

show_image(axes["A"], "network_schema.png")
show_image(axes["C"], "3d_reconstruction.png")
show_image(axes["E"], "fig1_plotE.png")

input_statistics(axes["B"])
fractional_statistics(axes["D"])


fig.savefig(args.save_destination+"fig1.pdf",  bbox_inches="tight")



