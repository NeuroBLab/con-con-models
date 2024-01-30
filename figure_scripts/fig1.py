import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from scipy.stats import sem
from scipy.stats import wilcoxon, mannwhitneyu
from ccmodels.plotting.utils import figure_saver, prepare_c1, prepare_d1


def plot_fig1(axes, cmap,
              plt_a: str, 
              plt_b: str,
              plt_c: list,
              plt_d: list,
              plt_e: str,
              ):
    
    ''' This function generates the plots making up Figure 1 of the paper 
    Inputs:
    axes: matplotlib axes to plot on
    plt_a: str, with the path of the image in figure a,
    plt_b: str, with the path of the image in figure b,
    plt_c: list, where each item is a dataframe with the information for the required counts of the sample of neuronal connectivity
    plt_d: list,where each item is  a dataframe of bootstrap samples of connetion proportions for neurons form L2/3 and L4
    plt_e: str, with the path of the image in figure e
    '''
    
    axs = axes.ravel()

    #A&B
    image1 = Image.open(plt_a)
    image2 = Image.open(plt_b)

    axs[0].imshow(image1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])


    axs[1].imshow(image2)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    #C
    axs[2].bar(['non-proofread', 'proofread'], [np.median(plt_c[0]['id']),np.median(plt_c[1]['id'])], color = ['red', 'green'])
    axs[2].set_title('Median input neurons')

    axs[3].bar(['non-proofread', 'proofread'], [np.median(plt_c[2]['id']),np.median(plt_c[3]['id'])], color = ['red', 'green'])
    axs[3].set_title('Median output neurons')


    #D
    axs[4].bar([0,1, 3,4], [np.mean(plt_d[0]['L2/3']),np.mean(plt_d[1]['L2/3']),np.mean(plt_d[0]['L4']), np.mean(plt_d[1]['L4'])], 
        yerr = [np.std(plt_d[0]['L2/3']),np.std(plt_d[1]['L2/3']),np.std(plt_d[0]['L4']), np.std(plt_d[1]['L4'])],
          tick_label = ['L2/3', 'L2/3_np', 'l4', 'l4_np'], color = ['purple', 'lightblue', 'darkorange', 'orange'])

    axs[4].set_xlabel('Layer', fontsize = 20)
    axs[4].set_ylabel('Mean input proportion', fontsize = 20)
    axs[4].tick_params(axis='both', which='major', labelsize=20)

    #E
    image3= Image.open(plt_e)
    axs[5].imshow(image3)
    axs[5].set_xticks([])
    axs[5].set_yticks([])
    axs[5].set_xticklabels([])
    axs[5].set_yticklabels([])


def main():
    '''Main function to plot and save figure 1'''

    #Defining Parser
    parser = argparse.ArgumentParser(description='''Generate plot for figure 1''')
    
    # Adding and parsing arguments
    parser.add_argument('width', type=int, help='Width of image (cm)')
    parser.add_argument('height', type=int, help='Height of image (cm)')
    parser.add_argument('save_destination', type=str, help='Destination path to save figure in')
    args = parser.parse_args()
    print('''
          
    Preparing data for the plot...
          
          ''')
    onproof_inputs_counts, proof_inputs_counts, nonproof_outputs_counts, proof_outputs_counts = prepare_c1()
    boots_propl_proof, boots_propl_noproof = prepare_d1()

    fig, axes = plt.subplots(nrows=3, ncols = 2, constrained_layout=True)#, figsize=style.two_col_size(height=12))

    print('''
          
    Plotting data
          
          ''')
    
    plot_fig1(axes, plt.cm.Blues, plt_a='images/network_schema.png', plt_b='images/3d_reconstruction.png', 
              plt_c=[onproof_inputs_counts, proof_inputs_counts, nonproof_outputs_counts, proof_outputs_counts],
               plt_d = [boots_propl_proof, boots_propl_noproof],
                plt_e ='images/fig1_plotE.png')


    print('''
          
    Saving plot...
          
          ''')
    
    figure_saver(fig, 'fig1', args.width, args.height, args.save_destination)

    print('''
          
    Plot saved!
      
          ''')

if __name__ == '__main__':
    main()