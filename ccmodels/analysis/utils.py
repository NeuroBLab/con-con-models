#Imports
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
from ccmodels.preprocessing.utils import constrainer

def sig_cross_table(data, test, test_var, grouping_var):
    '''generates a cross table for significances between variables in data in the same grouping variables and
    utilising the specified significance test. 
    In the usage for MICrONS data the grouping variable is the centred direction difference of the stimulus
    at that specific activity value
    
    Parameters:
    data: pandas DataFrame with the values to generate across table
    test: statistical test to apply
    grouping_var: string with name of column in data containing the variable to group by
    
    Returns:
    sig_crosstab: Pandas DataFrame in a cross tab format with significances between each of the grouping variable's
    values

    ''' 

    test_significances = defaultdict(list)
    test_significances_numeric = defaultdict(list)

    to_iterate = sorted(list(set(data[grouping_var])))
    iterated = []
    for key in sorted(to_iterate):
        for val in sorted(to_iterate):
            if val in iterated:
                test_significances[key].append('0')
                test_significances_numeric[key].append(1)
                continue
            else:
                if key == val:
                    test_significances[key].append('not_sig')
                    test_significances_numeric[key].append(1)
                else:
                    stat, pval = test(data[data[grouping_var] == key][test_var],data[data[grouping_var] == val][test_var])
                    direction = '-'
                    if np.median(data[data[grouping_var] == key][test_var])>np.median(data[data[grouping_var] == val][test_var]):
                        direction = '+'

                    if direction == '+':
                        test_significances_numeric[key].append(pval)
                    else:
                        test_significances_numeric[key].append(-pval)

                    n = len(to_iterate)
                    if pval<(0.001/((n*n)-n)):
                        test_significances[key].append(f'{direction} ***')
                    elif pval<(0.01/((n*n)-n)):
                        test_significances[key].append(f'{direction} **')
                    elif pval<(0.05/((n*n)-n)):
                        test_significances[key].append(f'{direction} *')
                    else:
                        test_significances[key].append('not_sig')
        iterated.append(key)
    
    sig_crosstab = pd.DataFrame(data = list(test_significances.values()), columns = list(test_significances.keys()), index = list(test_significances.keys()))
    sig_crosstab.index.name = grouping_var

    sig_crosstab_numeric = pd.DataFrame(data = list(test_significances_numeric.values()), columns = list(test_significances_numeric.keys()), index = list(test_significances_numeric.keys()))
    sig_crosstab_numeric.index.name = grouping_var

    return sig_crosstab, sig_crosstab_numeric

def zero_center_and_shift(pre_df, directions_col, pre_po_col, delta_ori_shifts, currents = True):
    '''This function centers the tuning curve of a presynaptic neuron at 0 (i.e. its maximum is at zero) 
    and then shifts the tuning curve by a specified delta to generate a new synthetic tuning curve with 
    respect to a syntehetic post synaptic neuron.
    The discretized directions shown in the stimulus are mapped from the [-2pi, 2pi]
    range to the [-pi, pi] range
    
    Parameters:    
    pre_df: data frame containing activities of pre-synaptic cell, 
    the neuron's preferred orientation (for centerning tuning curve), and the delta with which to shift the curve

    directions_col: str, name of column containing the directions of the stimulu shown at each activity value

    
    pre_po_col: str, name of the column containing presynaptic cell's preferred orientation

    delta_ori_shifts: list, containing delta by which to shift the column by
    
    Returns:
    reordered_act: list where each item is an array of the activity for a pre_synaptic cell
    with values reordered according to their new [-pi, pi] range

    constrained_dirs: list of directions remapped in range (-pi, pi]
    '''

    
    #center the presynaptic tuning curve at 0
    arr_diffs = pre_df[directions_col]-pre_df[pre_po_col].values
    
    #Shift it by the extracted delta ori
    #arr_diffs = arr_diffs-pre_df[delta_shift_col].values

    reordered_act = list()
    constrained_dirs = list()

    # post_center = list()

    # dirs_center = list()

    for i in range(len(arr_diffs)):

        #constrainn directions between (-pi, pi]
        all_tcentered = constrainer(list(arr_diffs)[i])
        #all_tcentered = np.around(all_truncated, 6)

        #Shift it by the extrcted delta ori
        #print(delta_ori_shifts[i])
        all_truncated = all_tcentered -  delta_ori_shifts[i] #-1.571 #pre_df[delta_shift_col].values[i]

        all_truncated = constrainer(all_truncated)
        all_truncated = np.around(all_truncated, 6)
        all_truncated[all_truncated <= -3.14] = 3.141593 #-3.141593

        #extract index sorted from smallest direction to largest
        idx= np.argsort(all_truncated)
    
        #generate array with activities of pre_synaptic cells
        if currents:
            activities = np.array(pre_df['current'].values[i].tolist())
        else:
            activities = np.array(pre_df['pre_activity'].values[i].tolist())


        #order these activities accoridng to their sorted value in the new
        #[-np.pi, np.pi] range
        reordered_act.append(list(activities[idx]))

        constrained_dirs.append(list(all_truncated[idx]))

        # post_center.append(list(activities[np.argsort(all_tcentered)]))

        # dirs_center.append(all_tcentered)


    return reordered_act, constrained_dirs 


def untuned_shifter(pre_df, directions_col, delta_ori_shifts, currents = True):
    '''This function centers the tuning curve of a presynaptic neuron at 0 (i.e. its maximum is at zero) 
    and then shifts the tuning curve by a specified delta to generate a new synthetic tuning curve with 
    respect to a syntehetic post synaptic neuron.
    The discretized directions shown in the stimulus are mapped from the [-2pi, 2pi]
    range to the [-pi, pi] range
    
    Parameters:    
    pre_df: data frame containing activities of pre-synaptic cell, 
    the neuron's preferred orientation (for centerning tuning curve), and the delta with which to shift the curve

    directions_col: str, name of column containing the directions of the stimulu shown at each activity value

    
    pre_po_col: str, name of the column containing presynaptic cell's preferred orientation

    delta_ori_shifts: list, containing delta by which to shift the column by
    
    Returns:
    reordered_act: list where each item is an array of the activity for a pre_synaptic cell
    with values reordered according to their new [-pi, pi] range

    constrained_dirs: list of directions remapped in range (-pi, pi]
    '''

    
    #center the presynaptic tuning curve at 0
    arr_diffs = pre_df[directions_col].values
    
    #Shift it by the extracted delta ori

    reordered_act = list()
    constrained_dirs = list()

    for i in range(len(arr_diffs)):

        #constrainn directions between (-pi, pi]
        all_tcentered = constrainer(list(arr_diffs)[i])

        #Shift it by the extrcted delta ori
        all_truncated = all_tcentered -  delta_ori_shifts[i] 

        all_truncated = constrainer(all_truncated)
        all_truncated = np.around(all_truncated, 6)
        all_truncated[all_truncated <= -3.14] = 3.141593 #-3.141593

        #extract index sorted from smallest direction to largest
        idx= np.argsort(all_truncated)
    
        #generate array with activities of pre_synaptic cells
        if currents:
            activities = np.array(pre_df['current'].values[i].tolist())
        else:
            activities = np.array(pre_df['pre_activity'].values[i].tolist())


        #order these activities accoridng to their sorted value in the new
        #[-np.pi, np.pi] range
        reordered_act.append(list(activities[idx]))

        constrained_dirs.append(list(all_truncated[idx]))


    return reordered_act, constrained_dirs

def tuning_encoder(connectome_subset, pre_col, post_col, label):
    '''Utility function to encode if pre and post neurons are tuned or not

    Args:
    connectome_subset: DF, with connectivity amongst neurons
    pre_col: str, name of column containing non-numerical label for PREsynaptic tuning property of neuron
    post_col: str, name of column containing non-numerical label for POSTsynaptic tuning property of neuron
    label: str, label defining an untuned neurone

    Returns: 
    connectome_subset: DF, same df passed as input but with two new columns (pre_tuned, post_tuned) showing 0 if neuron is
    untuned and 1 if it is tuned
    '''
    tuned_pre = []
    for i in connectome_subset[pre_col]:
        if i == label:
            tuned_pre.append(0)
        else:
            tuned_pre.append(1)

    tuned_post= []
    for i in connectome_subset[post_col]:
        if i == label:
            tuned_post.append(0)
        else:
            tuned_post.append(1)

    connectome_subset['pre_tuned'] = tuned_pre
    connectome_subset['post_tuned'] = tuned_post

    return connectome_subset

def tuning_segmenter(connectome_subset):
    ''' Utility function to segment a L2/3 and L4 connectome in the permutations of available subsets of tuned neuron-neuron connections'''
    #L4 -> L2/3
    l4t_l23t =  connectome_subset[(connectome_subset['post_tuned'] == 1) & (connectome_subset['pre_tuned'] == 1)
                            & (connectome_subset['pre_layer'] == 'L4')]

    l4t_l23u =  connectome_subset[(connectome_subset['post_tuned'] == 0) & (connectome_subset['pre_tuned'] == 1)
                            & (connectome_subset['pre_layer'] == 'L4')]

    l4u_l23u = connectome_subset[(connectome_subset['post_tuned'] == 0) & (connectome_subset['pre_tuned'] == 0)
                            & (connectome_subset['pre_layer'] == 'L4')]

    l4u_l23t = connectome_subset[(connectome_subset['post_tuned'] == 1) & (connectome_subset['pre_tuned'] == 0)
                            & (connectome_subset['pre_layer'] == 'L4')]
    
    l4_combinations = [l4t_l23t,l4t_l23u,l4u_l23u,l4u_l23t]

    #L2/3 -> L2/3
    l23t_l23t =  connectome_subset[(connectome_subset['post_tuned'] == 1) & (connectome_subset['pre_tuned'] == 1)
                            & (connectome_subset['pre_layer'] == 'L2/3')]

    l23t_l23u =  connectome_subset[(connectome_subset['post_tuned'] == 0) & (connectome_subset['pre_tuned'] == 1)
                            & (connectome_subset['pre_layer'] == 'L2/3')]

    l23u_l23u =  connectome_subset[(connectome_subset['post_tuned'] == 0) & (connectome_subset['pre_tuned'] == 0)
                            & (connectome_subset['pre_layer'] == 'L2/3')]

    l23u_l23t =  connectome_subset[(connectome_subset['post_tuned'] == 1) & (connectome_subset['pre_tuned'] == 0)
                            & (connectome_subset['pre_layer'] == 'L2/3')]
    
    l23_combinations = [l23t_l23t,l23t_l23u,l23u_l23u, l23u_l23t]
    
    return l4_combinations, l23_combinations

def compute_cur_delta_dist(connectome_subset):
    '''utility function to compute, for each single input of a postsynaptic neuron, the difference in current at the post synaptic 
    preferred orientation and the orthogonal orientation '''

    curdelta = []
    #Iterate through the postsynaptic preferred orientations
    for dirs, postPO, current in tqdm(zip(connectome_subset['pre_orientations'].values,connectome_subset['post_po'].values,
                        connectome_subset['current'].values ), total=connectome_subset.shape[0]):

        #Identify index of post PO
        idx_postPO = np.where(dirs.round(6) == postPO.round(6))[0]

        #Identify postsynpatic least PO
        if postPO < np.pi/2:
            leastPO = postPO+(np.pi/2)
        else:
            leastPO = postPO-(np.pi/2)

        #identify index of postsynpatic least PO
        idx_postLPO = np.where(dirs.round(6) == leastPO.round(6))[0]

        #Extract values or presynaptic input from a single neu at least and preferred orientations
        curPO = current[idx_postPO]
        curLPO = current[idx_postLPO]

        #Calculate the difference
        delta = curPO-curLPO
        curdelta.append(delta[0])
    
    return curdelta


# ------------------------------------
# --- WORKING WITH CURRENT -----------
# ------------------------------------

"""
def normalize_current(current, norm):
    #return (current-np.min(current))/(np.max(current)-np.min(current))
    return current / norm 
"""

def get_current_normalization(all_currents, mode="mean"):
    """
    Computes a constant to normalize all currents to. 
    It can do it to the max, so all currents are smaller than 1, or to the mean current
    mode can be eithe "mean" or "max". All_currents gives the table of currents to normalize from. 
    """
    if mode=="max":
        maxcurr = 0.0
        #Each element in the row is a list, so we have to work it out
        for curr in all_currents:
            maxcurr_neuron = np.max(curr)
            if  maxcurr_neuron > maxcurr:
                maxcurr = maxcurr_neuron
        return maxcurr
    elif mode=="mean":
        meancurr = 0.0
        for curr in all_currents:
            meancurr += curr.sum()
        return meancurr / (len(curr)*len(all_currents)) 



if __name__ == '__main__':
    print(f'cwd: {os.getcwd()}')