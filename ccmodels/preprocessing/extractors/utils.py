#Imports
import numpy as np
import pandas as pd 
from tqdm.auto import tqdm
from caveclient import CAVEclient
from standard_transform import minnie_transform_vx




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

def constrain_act_range(post_root_col, post_root_id, directions, pre_df, currents = True):
    '''This function maps the discretized directions shown in the stimulus from the [-2pi, 2pi]
    range to the [-pi, pi] range and re-orders the activities of each pre-synaptic
    connections of a specified post-synaptic cell according to the new direction mapping
    
    Parameters:
    post_root_col: str, column containing postsynaptic ids of neurons
    post_root_id: id of the post_synaptic cell
    directions: array of discretized directions in [-2pi, 2pi] range
    pre_df: data frame containing activities of pre-synaptic cell and key (post_root_id) specifiying which post_synaptic cell they connect to 
    
    Returns:
    reordered_act: list where each item is an array of the activity for a pre_synaptic cell
    with values reordered according to their new [-pi, pi] range

    constrained_dirs: list of directions remapped in range (-pi, pi]
    '''

    #select all pre synaptic cells
    cell = pre_df[pre_df[post_root_col] == post_root_id]
    
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
    for i in tqdm(range(0, postsynaptic_set.shape[0], neurs_per_steps), desc = 'Extracting connectome subset'):
        
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

def unique_neuronal_inputs(pt_root_id, neurons, client):
    '''function to extract all the unique neuronal inputs for a postsynaptic cell
    neurons: set of ids  of cells that are neurons, utilise the nucleus_neuron_svm table from Minnie65 v343 '''

    input_df = client.materialize.synapse_query(post_ids = pt_root_id)
    input_df = input_df.drop_duplicates(subset = 'pre_pt_root_id')
    neuronal_inputs = input_df[input_df['pre_pt_root_id'].isin(neurons)]

    return pd.DataFrame(neuronal_inputs)

def unique_neuronal_outputs(pt_root_id, neurons, client):
    '''function to extract all the unique neuronal outputs for a postsynaptic cell
     neurons: set of ids  of cells that are neurons, utilise the nucleus_neuron_svm table from Minnie65 v343'''

    output_df = client.materialize.synapse_query(pre_ids = pt_root_id)
    output_df = output_df.drop_duplicates(subset = 'post_pt_root_id')
    neuronal_outputs = output_df[output_df['post_pt_root_id'].isin(neurons)]

    return pd.DataFrame(neuronal_outputs)



def layer_extractor(input_df, transform, column = 'pre_pt_position'):
    input_df['pial_distances'] = transform.apply(input_df[column])

    #Use the y axis value to assign the corresponding layer as per Ding et al. 2023
    layers = []
    for i in input_df['pial_distances'].iloc[:]:
        if 0<i[1]<=98:
            layers.append('L1')
        elif 98<i[1]<=283:
            layers.append('L2/3')
        elif 283<i[1]<=371:
            layers.append('L4')
        elif 371<i[1]<=574:
            layers.append('L5')
        elif 574<i[1]<=713:
            layers.append('L6')
        else:
            layers.append('unidentified')

    input_df['cortex_layer'] = layers   
    return input_df


def subset_v1l234(client, table_name = 'coregistration_manual_v3', area_df = 'con-con-models/data_full/v1_n.csv'):
    '''This function takes a table of functionally matched neurons from the MICrONs connectomics database
    and returns a subset only containing neurons belowning to L2/3/4 of V1
    

    Args:
    client: CAVEclient needed to access MICrONS connectomics data
    table_name: name of table in CAVEClient database with functionally matched neurons
    area_df: DataFrame containing brain area of all neurons in functional database, uniquely identifiable
    by their (session, scan_idx, unit_id) tuples.

    Returns:
    v1l234_neur: pd.DataFrame only containing neurons from L2/3/4 of V1

    '''
    funct_match = client.materialize.query_table(table_name)
    funct_match_clean = funct_match[['pt_root_id', 'id', 'session', 'scan_idx', 'unit_id', 'pt_position']]
    
    v1_area = pd.read_csv(area_df)

    v1_neurons = funct_match_clean.merge(v1_area, on = ['session', 'scan_idx', 'unit_id'], how = 'inner')

    tform_vx = minnie_transform_vx()

    v1_neurons_layers = layer_extractor(v1_neurons, tform_vx, column = 'pt_position')

    v1l234_neur = v1_neurons_layers[v1_neurons_layers['cortex_layer'].isin(['L2/3', 'L4'])]

    return v1l234_neur
    

def connectome_feature_merger(connectome, neuron_features, pre_id = 'pre_pt_root_id', 
                        post_id = 'post_pt_root_id', neuron_id ='pt_root_id', conn_str = 'size' ):
    '''utility function to merge a connectome subset with a dataframe of neurons containing 
    features describing each neurone in the connectome (ex. selectivity, layer they belong to...)
    
    Args:
    connectome: df, subset of the connectome of interest, 
    neuron_features: df, with the features of interest for the neurons in the connectome
    pre_id: str, column name with the ids of the presynaptic neurons on which to match the features df on 
    post_id: str,column name with the ids of the postsynaptic neurons on which to match the features df on  
    neuron_id: str, column name in the features df identifying neuron on which to match on in the connectome
    conn_str: str, column name containing connection strength in the connectome df

    Returns:
    connectome_full: df, with the connectome subset and features for the pre and post neurons
    '''

    connectome = connectome.copy()
    neuron_features = neuron_features.copy()

    keep_same = [pre_id, post_id, conn_str, neuron_id]

    #Merge presynaptic data
    connectome_pre = connectome.merge(neuron_features, left_on = pre_id, 
                                      right_on = neuron_id, how = 'left', 
                                      suffixes = ('_pre', '_feat'))
    connectome_pre = connectome_pre[connectome_pre.columns.drop(list(connectome_pre.filter(regex='_feat')))]

    #remove repeated root id column
    connectome_pre.drop(columns = neuron_id, inplace = True)
    
    #Rename columns to highlight they identify presynaptic information
    connectome_pre.columns = ['{}{}'.format('' if c in keep_same else 'pre_', c) for c in connectome_pre.columns]

    #Merge postsynaptic data
    #Rename columns to highlight they identify postynaptic information
    neuron_features.columns = ['{}{}'.format('' if c in keep_same else 'post_', c) for c in neuron_features.columns]

    connectome_full = connectome_pre.merge(neuron_features, left_on = post_id, 
                                           right_on = neuron_id, how = 'left')
    
    #remove repeated root id column
    connectome_full.drop(columns = neuron_id, inplace = True)
    
    return connectome_full


def proofread_neurons(client, table, dendrites = True, axons = True):
    '''
    Identify and extract  proofread neurons
    Args:
    client: CAVEclient
    table: str, name of cave table to query
    dendrites: bool, whether to extract neurons with fully proofread dendrites
    axons: bool, whether to extract neurons with fully proofread axons
    
    Returns:
    proofread_neur: DF with information on proofread neurons
    '''

    # Set of fully proofread neurons
    proofread_neur = client.materialize.query_table(table)

    if dendrites:
        proofread_neur = proofread_neur[(proofread_neur['status_dendrite'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])]
    
    elif axons:
        proofread_neur = proofread_neur[(proofread_neur['status_axon'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])]

    elif dendrites and axons:
        proofread_neur = proofread_neur[(proofread_neur['status_dendrite'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])&
                                (proofread_neur['status_axon'] ==  'extended')]
    return proofread_neur

def client_version(version = 343):
    '''Define the version of the CAVE client database you want to access and use for the analysis'''
    
    client = CAVEclient('minnie65_public')
    client.materialize.version = version
    
    return client