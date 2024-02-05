#Imports
import numpy as np
import pandas as pd 
from tqdm.auto import tqdm


def min_act(max_rad, model_type, dirs):
    '''This function returns the oreintation for where the minimum of the selective activity should be
    
    Parameters:
    max_rad: integer or float with estimated preferred orientation of the cell
    model_type: string idenfiying whether the modelled cell is  oreintationn selectivity (model_type = 'single')
    or orientation and direction selectivity (model_type = 'double')
    
    Returns:
    min_rad: estimated least preferred orientation
    '''
    #If there is a single peak, frequency of 2pi -> neuron is direction selective
    if model_type == 'direction':
        if max_rad>np.pi:
            min_rad = max_rad-np.pi
        else:
            min_rad = max_rad+np.pi
    
    #If there are two peak, frequency of pi -> neuron is orientation selective
    # NOTE: here we treat those neurons that are not selective as orientation selective 
    # for the purpose of calculating an osi value also for them

    else:
        if max_rad>(np.pi*1.5):
            min_rad = max_rad-(np.pi/2)
        else:
            min_rad = max_rad+(np.pi/2)

    
    ind_min = np.argmin(np.abs(dirs- min_rad))

    closemin = dirs[ind_min]
    return closemin

def constrainer(dirs, reversed = False):
    '''Function that constrains given matrix of directions between [-2pi, 2pi] in to (-pi, pi]
    
    Parameters:
    dirs: numpy array of directions
    
    Returns:
    all_truncated: numpy array of constrained directions
    '''
    

    #remap between [-np.pi, np.pi]
    #find cells below -np.pi
    smaller = (dirs<=-np.pi).astype(int)*(2*np.pi)
    
    #find cells above np.pi
    larger = (dirs>np.pi).astype(int)*(2*np.pi)

    #add 2pi to dirs below -np.pi
    small_truncated = dirs+smaller

    #subtract 2pi to cells above np.pi
    all_truncated = small_truncated-larger

    if reversed:
        smaller = (dirs<0).astype(int)*(2*np.pi)
        detruncated = dirs+smaller
        return detruncated

    return all_truncated

def constrain_act_range(post_root_id, directions, pre_df, currents = True):
    '''This function maps the discretized directions shown in the stimulus from the [-2pi, 2pi]
    range to the [-pi, pi] range and re-orders the activities of each pre-synaptic
    connections of a specified post-synaptic cell according to the new direction mapping
    
    Parameters:
    post_root_id: id of the post_synaptic cell
    directions: array of discretized directions in [-2pi, 2pi] range
    pre_df: data frame containing activities of pre-synaptic cell and key (post_root_id) specifiying which post_synaptic cell they connect to 
    
    Returns:
    reordered_act: list where each item is an array of the activity for a pre_synaptic cell
    with values reordered according to their new [-pi, pi] range

    constrained_dirs: list of directions remapped in range (-pi, pi]
    '''

    #select all pre synaptic cells
    cell = pre_df[pre_df['post_id'] == post_root_id]
    
    #differences with post max
    arr_diffs = directions-cell['post_po'].values[0]

    #constrainn directions between (-pi, pi]
    all_truncated = constrainer(arr_diffs)
    all_truncated = np.around(all_truncated, 6)
    all_truncated[all_truncated ==-3.141593] = 3.141593

    #extract index sorted from smallest direction to largest
    idx= np.argsort(all_truncated)
    
    #generate array with activities of pre_synaptic cells
    if currents:
        activities = np.array(cell['current'].tolist())
    else:
        activities = np.array(cell['pre_activity'].tolist())


    #order these activities accoridng to their sorted value in the new
    #[-np.pi, np.pi] range
    reordered_act = list(activities[:,idx])

    constrained_dirs = list(all_truncated[idx])


    return reordered_act, constrained_dirs


def connectome_constructor(client, presynaptic_set, postsynaptic_set, neurs_per_steps = 500):
    '''
    Function to construct the connectome subset for the neurons specified in the presynaptic_set and postsynaptic_set.

    Args:
    client: CAVEclient needed to access MICrONS connectomics data
    presynaptic_set: 1-d array of non repeated root_ids of presynaptic neurons for which to extract postsynaptoc connections in postynaptic_set
    postynaptic_set: 1-d array of non repeated root_ids of postsynaptic neurons for which to extract presynaptic connections in presynaptic_set
    neurs_per_steps: number of postsynaptic neurons for which to recover presynaptic connectivity per single call to the connectomics
        database. Since the connectomics database has a limit on the number of connections you can query at once
        this iterative method optimises querying multiple neurons at once, as opposed to each single neuron individually,
        while also preventing the queries from crashing. I have tested that for a presynaptic set of around 8000 neurons
        you can reliably extract the connectivity for around 500 postsynaptic neurons at a time.
    '''
    
    if_thresh = (postsynaptic_set.shape[0]//neurs_per_steps)*neurs_per_steps
    
    syndfs = []
    for i in tqdm(range(0, postsynaptic_set.shape[0], neurs_per_steps)):
        
        if i <if_thresh:
            post_ids = postsynaptic_set[i:i+neurs_per_steps]

        else:
            post_ids = postsynaptic_set[i:]

        sub_syn_df = client.materialize.query_table('synapses_pni_2',
                                            filter_in_dict={'pre_pt_root_id': presynaptic_set,
                                                            'post_pt_root_id':post_ids})
            
        syndfs.append(np.array(sub_syn_df[['pre_pt_root_id', 'post_pt_root_id', 'size']]))
    
    syn_df = pd.DataFrame({'pre_pt_root_id':np.vstack(syndfs)[:, 0], 'post_pt_root_id': np.vstack(syndfs)[:, 1], 'size': np.vstack(syndfs)[:, 2]})
    return syn_df

def func_pre_subsetter(client, to_keep, func_id):
    '''This is a function to return all of the pre_synaptic neurons that are functionally matched for a given 
    neuron id passed as func_id

    Args: 
    client: CAVEclient needed to access MICrONS connectomics data
    to_keep: set of root ids of functionally matched neurons that we wish to subset synapse df of func_id by
    func_id: root_id of neuorn for which we want to extract pre synaptic connections.

    Returns: 
    df containing connections between func_id neuron and all its functionally matched pre synaptic neurons
    '''

    syn = client.materialize.synapse_query(post_ids=func_id)
    sub = syn[syn['pre_pt_root_id'].isin(to_keep)].loc[:, ['post_pt_root_id', 'pre_pt_root_id', 'size','post_pt_position', 'pre_pt_position']]
    return sub