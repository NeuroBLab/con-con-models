import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.stats import sem
from scipy.stats import wilcoxon, mannwhitneyu
from ccmodels.analysis.utils import tuning_encoder
from ccmodels.plotting.utils import figure_saver, prepare_b2, prepare_c2, prepare_d2, prepare_e2




def plot_fig2(axes, cmap, norm_const, 
              plt_a: pd.DataFrame, 
              plt_b: list,
              plt_c: list,
              plt_d: list,
              plt_e: list,
              ):

    ''' This function generates the plots making up Figure 2 of the paper 
    Inputs:
    axes: matplotlib axes to plot on
    norm_const: normalising constant,
    plt_a: DF, with the data with also OSI information on all V1 neurons of interest,
    plt_b: list, containing two lists each with the bootstrap samples of connetion probabilities 
        according to pre and postsynpatic tuning
    plt_c: list, containing two dataframes with the boostrap on the connection probabilities 
        as a function of the difference in preferredo rientation, the first for 
        L2/3 neurons the second for L4 neurons
    plt_d: list,containing two lists each with the distirbutions of connection strengths according 
        to pre and postsynpatic tuning
    plt_e: list, with four sub lists containing the cumulative counts and the bins of connections strengths for a specific 
        pre-post difference in preferred orientation
    '''
    normcstr = norm_const
    axs = axes.ravel()

    # Plot A
    l23_n = plt_a[plt_a['cortex_layer'] == 'L2/3']
    l4_n = plt_a[plt_a['cortex_layer'] == 'L4']
    bins = np.linspace(0, 1,20)

    x1 = l23_n[(l23_n['model_type'] == 'orientation') | (l23_n['model_type'] == 'direction')]['osi']
    histl23, binsl23 = np.histogram(x1, bins)
    histl23n = histl23/np.sum(histl23)
    axs[0].step(binsl23[:-1], histl23n,
            color = 'purple')
    axs[0].axvline(x =binsl23[0], ymax = histl23n[0]/np.max(histl23n), color = 'purple')
    axs[0].axvline(x =binsl23[-2], ymax = (histl23n[-1]/np.max(histl23n))-0.01, color = 'purple')


    axs[0].set_ylabel('% Neurons')
    axs[0].set_xlabel('OSI L2/3 neurons')
    axs[0].set_ylim(bottom = 0)



    x2 = l4_n[(l4_n['model_type'] == 'orientation') | (l4_n['model_type'] == 'direction')]['osi']
    histl4, binsl4 = np.histogram(x2, bins)
    histl4n = histl4/np.sum(histl4)

    axs[1].step(binsl4[:-1], histl4n,
            color = 'darkorange')
    axs[1].axvline(x =binsl4[0], ymax = histl4n[0]/np.max(histl4n), color = 'darkorange')
    axs[1].axvline(x =binsl4[-2], ymax = (histl4n[-1]/np.max(histl4n))-0.001, color = 'darkorange')

    axs[1].set_ylabel('% Neurons')
    axs[1].set_xlabel('OSI L4 neurons')
    axs[1].set_ylim(bottom = 0)

    #Plot B
    bars_dashed = [1,3,5,7]
    bars = axs[2].bar([0,1, 3,4, 6,7, 9,10], 
            [np.mean(plt_b[0][0]), np.mean(plt_b[0][1]),  np.mean(plt_b[0][2]), np.mean(plt_b[0][3]),  
             np.mean(plt_b[1][0]),np.mean(plt_b[1][1]), np.mean(plt_b[1][2]),np.mean(plt_b[1][3])],
            yerr= [np.std(plt_b[0][0]), np.std(plt_b[0][1]),  np.std(plt_b[0][2]), np.std(plt_b[0][3]), 
                    np.std(plt_b[1][0]),np.std(plt_b[1][1]), np.std(plt_b[1][2]),np.std(plt_b[1][3])],
            edgecolor = ['purple', 'purple', 'purple', 'purple', 'darkorange', 'darkorange',
                         'darkorange', 'darkorange'], fill=False)
    for i in bars_dashed:
        bars[i].set_linestyle('--')

    axs[2].set_xticks([0.5,3.5,6.5,9.5], ['L2/3 tuned', 'L2/3 untuned', 'L4 tuned', 'L4 untuned'])
    axs[2].set_xlabel('Input', fontsize = 20)
    axs[2].set_ylabel('Connection Probability', fontsize = 20)
    #axs.set_title('L2/3 and L4 Input proportion', fontsize = 20)
    axs[2].tick_params(axis='both', which='major', labelsize=12)


    legend_elements = [plt.Line2D([0], [0], linestyle='-', color='black', label='Tuned output'),
                    plt.Line2D([0], [0], linestyle='--', color='black', label='Untuned output')]


    # Add the legend to the plot
    axs[2].legend(handles=legend_elements, loc=(0.21,0.85))


    #Plot C
    axs[3].fill_between(plt_c[0]['directions'],plt_c[0]['mean']-plt_c[0]['std'],
                 plt_c[0]['mean']+plt_c[0]['std'], color = 'purple', alpha = 0.2)
    axs[3].plot(plt_c[0]['directions'],plt_c[0]['mean'], color = 'purple',
            label = 'L2/3')
    axs[3].scatter(plt_c[0]['directions'],plt_c[0]['mean'], color = 'black', zorder = 3)



    axs[3].fill_between(plt_c[1]['directions'],plt_c[1]['mean']-plt_c[1]['std'],
                    plt_c[1]['mean']+plt_c[1]['std'],color = 'darkorange',  alpha = 0.2)
    axs[3].plot(plt_c[1]['directions'],plt_c[1]['mean'], color = 'darkorange',
            label = 'L4')
    axs[3].scatter(plt_c[1]['directions'],plt_c[1]['mean'], color = 'black', zorder = 3)
    axs[3].set_ylabel('Connection Probability', fontsize = 20)
    axs[3].set_xlabel('∆ori', fontsize = 20)

    #axs.set_title('Connection probability at varying ∆ori', fontsize = 20)

    axs[3].tick_params(axis='both', which='major', labelsize=15)

    axs[3].legend(fontsize = 15, loc = 'upper right')

    # Significance band
    axs[3].annotate('***', xy=(0.8, 0.043), xytext=(0.8, 0.035), xycoords='data', 
                fontsize=15*1.5, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1', lw=2.0, color='k'))


    #Plot D

    bars_dashed = [1,3,5,7]
    bars = axs[4].bar([0,1, 3,4, 6,7, 9,10], 
                [np.mean(plt_d[0][0])/normcstr, np.mean(plt_d[0][1])/normcstr,  
                 np.mean(plt_d[0][2])/normcstr, np.mean(plt_d[0][3])/normcstr,  
                 np.mean(plt_d[1][0])/normcstr,np.mean(plt_d[1][1])/normcstr, 
                 np.mean(plt_d[1][2])/normcstr,np.mean(plt_d[1][3])/normcstr],
            yerr= [np.std(plt_d[0][0])/(normcstr * np.sqrt(len(plt_d[0][0]))), 
                   np.std(plt_d[0][1])/(normcstr * np.sqrt(len(plt_d[0][1]))),  
                   np.std(plt_d[0][2])/(normcstr * np.sqrt(len(plt_d[0][2]))), 
                   np.std(plt_d[0][3])/(normcstr * np.sqrt(len(plt_d[0][3]))),  
                   np.std(plt_d[1][0])/(normcstr * np.sqrt(len(plt_d[1][0]))),
                   np.std(plt_d[1][1])/(normcstr * np.sqrt(len(plt_d[1][1]))),
                   np.std(plt_d[1][2])/(normcstr * np.sqrt(len(plt_d[1][2]))),
                   np.std(plt_d[1][3])/(normcstr * np.sqrt(len(plt_d[1][3])))],
            edgecolor = ['purple', 'purple', 'purple', 'purple', 'darkorange', 'darkorange','darkorange', 'darkorange'], fill=False)
    
    for i in bars_dashed:
        bars[i].set_linestyle('--')

    axs[4].set_xticks([0.5,3.5,6.5,9.5], ['L2/3 tuned', 'L2/3 untuned', 'L4 tuned', 'L4 untuned'])
    axs[4].set_xlabel('Input', fontsize = 20)
    axs[4].set_ylabel('Connection Strength', fontsize = 20)
    #axs.set_title('L2/3 and L4 Input proportion', fontsize = 20)
    axs[4].tick_params(axis='both', which='major', labelsize=12)


    legend_elements = [plt.Line2D([0], [0], linestyle='-', color='black', label='Tuned output'),
                    plt.Line2D([0], [0], linestyle='--', color='black', label='Untuned output')]

    # Add the legend to the plot
    axs[4].legend(handles=legend_elements, loc=(0.21,0.85))


    #plot E
    axs[5].step(plt_e[0][1], plt_e[0][0]/np.sum(plt_e[0][0][-1]), color = 'darkorange', label = 'L4: 0 rads')
    axs[5].step(plt_e[1][1], plt_e[1][0]/np.sum(plt_e[1][0][-1]), color = 'gold', label = 'L4: pi/2 rads')
    axs[5].step(plt_e[2][1], plt_e[2][0]/np.sum(plt_e[2][0][-1]), color = 'purple', label = 'L2/3: 0 rads')
    axs[5].step(plt_e[3][1], plt_e[3][0]/np.sum(plt_e[3][0][-1]), color = 'violet',label = 'L2/3: pi/2 rads')
    axs[5].legend()



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
    
    v1_neurons = pd.read_pickle('../con-con-models/data/v1l234_neurons.pkl')
    v1_connections = pd.read_pickle('../con-con-models/data/v1l234_connections.pkl')
    #Encoding numerically if input and output is tuned or untuned
    v1_connections = tuning_encoder(v1_connections,'pre_type', 'post_type', 'not_selective')
    normcstr = np.mean(v1_connections[v1_connections['pre_layer'] == 'L2/3']['size'])

    l23_tuning_connp, l4_tuning_connp = prepare_b2(v1_connections)
    l23_boots, l4_boots = prepare_c2(v1_connections)
    l23_comb_strengths, l4_comb_strengths = prepare_d2(v1_connections)
    cumul_dists = prepare_e2(v1_connections, [0, 1.570796, 0, 1.570796])

    fig, axes = plt.subplots(nrows=3, ncols = 2, constrained_layout=True)

    print('''
          
    Plotting data
          
          ''')
    
    plot_fig2(axes,plt.cm.Blues,normcstr,plt_a=v1_neurons, plt_b=[l23_tuning_connp, l4_tuning_connp],
              plt_c = [l23_boots, l4_boots], plt_d=[l23_comb_strengths, l4_comb_strengths],
               plt_e=cumul_dists)


    print('''
          
    Saving plot...
          
          ''')
    
    figure_saver(fig,'fig2', args.width, args.height, args.save_destination)

    print('''
          
    Plot saved!
      
          ''')

if __name__ == '__main__':
    main()