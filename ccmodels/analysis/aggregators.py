#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
from scipy import stats
from ccmodels.analysis.simulators import bootstrap_medians



def act_aggregator(acts_df, grouping_column, currents_column):
    '''This function aggregate the currents of neurons by a specified grouping variable
     Parameters:
      acts_df: pandas data frame object containing all the columns specified below
      grouping_column: string with name of the column with the grouping property
      currents_column: string with name of column containing the currents, each row item is a numpy array

    Returns:
    actJ: a dictionary where keys are each instance of the goruping column variable and the 
    values are the summed currents for that gruping variable instance
        '''
    
    #initiate dictionary
    actJ = defaultdict(list)

    #loop through maximum values and summ all of the weighted activities
    #of pre synaptic cells with that maximum
    for i in tqdm(set(acts_df[grouping_column].values)):
        count = acts_df[acts_df[grouping_column] == i].shape[0]
        act = sum(acts_df[acts_df[grouping_column] == i][currents_column].values)
        act_l = list(act)
        act_l.append(count)
        actJ[i] = act_l
    return actJ



def sep_difference_plotter(layer_data, dict_data, xnumb = 4, ynumb = 4, sharey = True, size = (30,20), title = ''):
    '''This function allows to plot the sum of currents of the pre synaptic cells centred at 0 and
    grouped by the difference between the pre and post synaptic preferred orientation difference
    
    Parameters:
    layer_data: Dataframe with information on presynaptic annd post synaptic angle differences
    dict_data: dictionary with summed activities for each pre and post preferred orientation
    xnumb, ynumb: number of rows and columns in the subplot
    sharey: share y axis between subplots
    size: size of plot
    title: title of plot

    Returns:
    figure consisting of subplots for each grouped angle value
    
    '''
    
    #Plotting the sum of the cell
    fig, axes = plt.subplots(xnumb, ynumb, figsize = size, sharey=sharey)

    for ax, rad in zip(axes.ravel(),sorted(dict_data.keys())):
        pdm = layer_data[layer_data['delta_ori_constrained'] == rad]['new_dirs'].values[0]
        ax.plot(pdm, dict_data[rad][:-1], label = 'summed activity')
        ax.axvline(0, color = 'red', label= 'disc maximum')
        ax.set_xlabel('shown_orientation-max_post')
        ax.set_ylabel('current')
        ax.set_title(f'∆ori: {round(rad,2)}, n pre_neurons: {dict_data[rad][-1]}')
        
    plt.suptitle(title, fontsize = 30,x = 0.5, y = 1.03 )
    plt.legend()
    plt.tight_layout()
    plt.show()




def all_inputs_aggregator(layer_data, activity_column, dirs_columns, grouping = 'mean', dir_range = 'full'):
    '''This function allows to group all the activity of all presynaptic neurons and returns both the sum,
    the average and an ungrouped version of the dataframe
    
    Parameters:
    layer_data: dataframe with activities for neurons of a specific layer
    activity_column: string with the name of the column containing an array per row with the 
    response of each neuron (1 neuron per row)
    dirs_columns: string with the name of the column containing an array per row with the 
    angles of the stimuli shown. 1 neuron per row.

    Returns:
    a_grouped: DataFrame with responses and normalised responses averaged across neurons grouped by orientations
    shown (columns with standard errors are also included)

    ungrouped: DataFrame with all of the responses (not averaged) across neurons
    
    '''
    
    #extract all the activity of the neurons and the corresponding directions in two big lists
    ndirs = []
    nacts = []
    norm = []
    for ndr, at in zip(layer_data[dirs_columns], layer_data[activity_column]):
        if dir_range == 'half':
            ndirs += list(np.abs(ndr))
        elif dir_range == 'full':
            ndirs+=list(ndr)

        #Normalising
        atnorm = (at-np.min(at))/(np.max(at)-np.min(at))
        nacts+= list(at)
        norm+= list(atnorm)


    #Save it in a data frame for plotting purposes
    ungrouped = pd.DataFrame({'dirs': ndirs, 'cur':nacts, 'norm_cur':norm})

    #Aggregating the data for plotting

    #Round the directions
    a = ungrouped.round({'dirs':6, 'cur':0, 'norm_cur':0})

    #sort values by direction
    asort = a.sort_values(by = 'dirs')

    #Group values by direction and sum the activities
    a_grouped = asort.groupby('dirs').sum().reset_index()
    a_grouped = a_grouped.rename(columns = {'cur':'sum_cur', 'norm_cur':'sum_normedcur'})

    if grouping == 'mean':
        #Average the activities
        a_grouped['avg_cur'] = asort.groupby('dirs').mean().reset_index()['cur']
        a_grouped['avg_norm_cur'] = asort.groupby('dirs').mean().reset_index()['norm_cur']

        #Calculate standard error
        a_grouped['cur_sem'] = asort.groupby('dirs')['cur'].sem().reset_index()['cur']
        a_grouped['norm_cur_sem'] = asort.groupby('dirs')['norm_cur'].apply(lambda x: stats.sem(x)).reset_index()['norm_cur']
    
    elif grouping == 'median':
        #Average the activities
        a_grouped['avg_cur'] = asort.groupby('dirs').median().reset_index()['cur']
        a_grouped['avg_norm_cur'] = asort.groupby('dirs').median().reset_index()['norm_cur']

        #Calculate standard error
        a_grouped['cur_sem'] = asort.groupby('dirs')['cur'].apply(lambda x: stats.sem(x)).reset_index()['cur']
        a_grouped['norm_cur_sem'] = asort.groupby('dirs')['norm_cur'].apply(lambda x: stats.sem(x)).reset_index()['norm_cur']

    return a_grouped, ungrouped



def all_inputs_aggregator2(layer_data, activity_column, dirs_columns, grouping = 'mean', dir_range = 'full'):
    '''This function allows to group all the activity of all presynaptic neurons and returns both the sum,
    the average and an ungrouped version of the dataframe
    
    Parameters:
    layer_data: dataframe with activities for neurons of a specific layer
    activity_column: string with the name of the column containing an array per row with the 
    response of each neuron (1 neuron per row)
    dirs_columns: string with the name of the column containing an array per row with the 
    angles of the stimuli shown. 1 neuron per row.

    Returns:
    a_grouped: DataFrame with responses and normalised responses averaged across neurons grouped by orientations
    shown (columns with standard errors are also included)

    ungrouped: DataFrame with all of the responses (not averaged) across neurons
    
    '''
    
    #extract all the activity of the neurons and the corresponding directions in two big lists
    ndirs = []
    nacts = []
    norm = []
    for ndr, at in zip(layer_data[dirs_columns], layer_data[activity_column]):
        if dir_range == 'half':
            ndirs += list(np.abs(ndr))
        elif dir_range == 'full':
            ndirs+=list(ndr)

        #Normalising
        atnorm = (at-np.min(at))/(np.max(at)-np.min(at))
        nacts+= list(at)
        norm+= list(atnorm)


    #Save it in a data frame for plotting purposes
    ungrouped = pd.DataFrame({'dirs': ndirs, 'cur':nacts, 'norm_cur':norm})

    #Aggregating the data for plotting

    #Round the directions
    a = ungrouped.round({'dirs':6, 'cur':0, 'norm_cur':0})

    #sort values by direction
    asort = a.sort_values(by = 'dirs')

    #Group values by direction and sum the activities
    a_grouped = asort.groupby('dirs').sum().reset_index()
    a_grouped = a_grouped.rename(columns = {'cur':'sum_cur', 'norm_cur':'sum_normedcur'})

    if grouping == 'mean':
        #Average the activities
        a_grouped['avg_cur'] = asort.groupby('dirs').mean().reset_index()['cur']
        a_grouped['avg_norm_cur'] = asort.groupby('dirs').mean().reset_index()['norm_cur']

        #Calculate standard error
        a_grouped['cur_sem'] = asort.groupby('dirs')['cur'].sem().reset_index()['cur']
        a_grouped['norm_cur_sem'] = asort.groupby('dirs')['norm_cur'].apply(lambda x: stats.sem(x)).reset_index()['norm_cur']
    
    elif grouping == 'median':
        #Average the activities
        a_grouped['avg_cur'] = asort.groupby('dirs').median().reset_index()['cur']
        a_grouped['avg_norm_cur'] = asort.groupby('dirs').median().reset_index()['norm_cur']

        #Calculate standard error
        a_grouped['cur_sem'] = asort.groupby('dirs')['cur'].apply(lambda x: bootstrap_medians(x)).reset_index()['cur']
        #a_grouped['norm_cur_sem'] = asort.groupby('dirs')['norm_cur'].apply(lambda x: bootstrap_medians(x)).reset_index()['norm_cur']

    return a_grouped, ungrouped, asort


def cumul_dist(data, n_bins):

    histv, bins = np.histogram(data, bins = n_bins)
    delta_bins = np.diff(bins)
    cumulhist = np.cumsum(histv)

    return cumulhist, bins[1:]