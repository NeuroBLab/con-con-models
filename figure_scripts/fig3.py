import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm.auto import tqdm
from PIL import Image
from ccmodels.plotting.utils import figure_saver, prepare_b3, prepare_c3, prepare_d3, prepare_e3


def plot_fig3(axes, cmap, 
              plt_a: pd.DataFrame, 
              plt_b: list,
              plt_c: np.array,
              plt_d: list,
              plt_e: list,
              ):

    ''' This function generates the plots making up Figure 2 of the paper 
    Inputs:
    axes: matplotlib axes to plot on
    norm_const: normalising constant,
    plt_a: str, path to png image,
    plt_b: list, containing average current for l23, proportion of inputs for l23, average current for l4, proportion
        of inputs from l4
    plt_c: np.array, containing distribution of delta currents at least and at preferred orientation
    plt_d: list,containing two arrays one with the values and one the bins of the tPO-oPO simulation
    plt_e: list, containing two lists with the in degree of a neuron values and the mses for the simulation with the 
        corresponding in degree value in the other list
    '''

    axs = axes.ravel()

    #Plot A
    image = Image.open(plt_a)

    axs[0].imshow(image)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])

    
    #Plot B
    norm_c = np.max((plt_b[0]['avg_cur']*plt_b[1]) + (plt_b[2]['avg_cur']*plt_b[3]))

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    current = (plt_b[0]['avg_cur']*plt_b[1]) + (plt_b[2]['avg_cur']*plt_b[3])
    yerror_t =np.sqrt((plt_b[0]['cur_sem']**2)+(plt_b[2]['cur_sem']**2))

    #Total Current
    axs[1].fill_between(plt_b[0]['dirs'], (current- yerror_t)/norm_c, 
                    (current + yerror_t)/norm_c,
                    color='green', alpha=0.2)
    axs[1].plot(plt_b[0]['dirs'], current/norm_c,
                 linewidth = 1, color = 'green', zorder = 2, label = 'Total current')
    axs[1].scatter(plt_b[0]['dirs'], current/norm_c,
    color = 'black', s = 5, zorder = 3)


    #L2/3
    axs[1].fill_between(plt_b[0]['dirs'], ((plt_b[0]['avg_cur']*plt_b[1]) - yerror_t)/norm_c, 
                    ((plt_b[0]['avg_cur']*plt_b[1]) + yerror_t)/norm_c,
                    color='purple', alpha=0.2)

    axs[1].plot(plt_b[0]['dirs'], plt_b[0]['avg_cur']*plt_b[1]/norm_c, 
                linewidth = 1, color = 'purple', label = 'L2/3 current')
    axs[1].scatter(plt_b[0]['dirs'], plt_b[0]['avg_cur']*plt_b[1]/norm_c, color = 'black', s = 5, zorder=3)


    #L4
    axs[1].fill_between(plt_b[0]['dirs'], ((plt_b[2]['avg_cur']*plt_b[3])- yerror_t)/norm_c, 
                    ((plt_b[2]['avg_cur']*plt_b[3]) + yerror_t)/norm_c,
                    color='darkorange', alpha=0.2)

    axs[1].plot(plt_b[2]['dirs'], plt_b[2]['avg_cur']*plt_b[3]/norm_c,
                 linewidth = 1, color = 'darkorange', label =' L4 current')
    axs[1].scatter(plt_b[2]['dirs'], plt_b[2]['avg_cur']*plt_b[3]/norm_c, color = 'black',s = 5, zorder=3)


    axs[1].set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25], [0, '', 0.5, '', 1, ''])

    axs[1].set_xticks([ -np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            [r'- $\pi$', r'- $\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    axs[1].tick_params(axis='both', which='major', labelsize=18)
    

    #Plot C
    bins=np.linspace(-20,20,100)

    axs[2].hist(plt_c,bins,density=True,cumulative=False,histtype='step',lw=2, 
            color = 'green', label = f'Mean: {round(np.mean(plt_c), 2)}');



    axs[2].set_ylabel('PDF', fontsize = 15)
    axs[2].margins(x=0)
    axs[2].set_xlabel('∆Current at 0 and π/2 orientations', fontsize = 15)
    axs[2].legend(fontsize = 12)
    axs[2].tick_params(axis='both', which='major', labelsize=15)

    #Plot D
    probvals = [0]+list(plt_d[0]/np.sum(plt_d[0]))
    axs[3].step(plt_d[1], probvals, color = 'green')
    axs[3].axvline(plt_d[1][-1],ymin = 0, ymax = 0.40, color = 'green')
    axs[3].set_ylim(0, 0.2)

    axs[3].set_xlabel('|tPO-oPO| (rad)', fontsize = 18)
    axs[3].set_ylabel('Probability', fontsize = 18)
    axs[3].tick_params(axis='both', which='major', labelsize=18)

    axs[3].set_yticks([0,0.05, 0.1, 0.15, 0.2], [0, '', 0.1, '', ''])

    axs[3].set_xticks([ 0, np.pi/2, np.pi],
            ['0', r'$\pi/2$', r'$\pi$'])
    

    #Plot E
    axs[4].plot(plt_e[0], plt_e[1], color = 'green')

    axs[4].tick_params(axis='both', which='major', labelsize=18)
    axs[4].set_ylabel('Neuron In degree', fontsize = 18)
    axs[4].set_xlabel('MSE', fontsize = 18)







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
    
    v1_connections = pd.read_pickle('../con-con-models/data/v1l234_connections.pkl')
    proofread_input_n = pd.read_csv('../con-con-models/data/proofread_l234_inputs.csv')

    #Encoding numerically if input and output is tuned or untuned
    normcstr = np.mean(v1_connections[v1_connections['pre_layer'] == 'L2/3']['size'])

    t23_grouped, propl23, t4_grouped, propl4 = prepare_b3(v1_connections, proofread_input_n)
    ltot_curdelta = prepare_c3(v1_connections)
    valsabs, binsabs = prepare_d3(v1_connections)
    ks, mses = prepare_e3(v1_connections)

    fig, axes = plt.subplots(nrows=3, ncols = 2, constrained_layout=True)

    print('''
          
    Plotting data
          
          ''')
    
    plot_fig3(axes,plt.cm.Blues,plt_a='images/fig3_plotA.png', 
              plt_b=[t23_grouped, propl23, t4_grouped, propl4], plt_c = ltot_curdelta, plt_d=[valsabs, binsabs],
               plt_e=[ks, mses])


    print('''
          
    Saving plot...
          
          ''')
    
    figure_saver(fig, 'fig3', args.width, args.height, args.save_destination)

    print('''
          
    Plot saved!
      
          ''')

if __name__ == '__main__':
    main()