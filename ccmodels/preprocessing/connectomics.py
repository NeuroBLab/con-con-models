import numpy as np
import pandas as pd
from tqdm import tqdm
from caveclient import CAVEclient
from standard_transform import minnie_transform_vx
from .utils import layer_extractor


def client_version(version = 343):
    '''Define the version of the CAVE client database you want to access and use for the analysis'''
    
    client = CAVEclient('minnie65_public')
    client.materialize.version = version
    
    return client

def load_table(client, name):
    '''Function to load a table from the CAVEclient database
    
    Args:
    client: CAVEclient needed to access MICrONS connectomics data
    name: str, name of the table to load
    
    Returns:
    df: pd.DataFrame with the table loaded from the database'''
    
    return client.materialize.query_table(name)

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


def subset_v1l234(client, table_name = 'coregistration_manual_v3', area_df = 'con-con-models/data/raw/area_membership.csv'):
    '''This function takes a table of functionally matched neurons from the MICrONs connectomics database
    and returns a subset only containing neurons belonging to L2/3/4 of V1
    

    Args:
    client: CAVEclient needed to access MICrONS connectomics data
    table_name: name of table in CAVEClient database with functionally matched neurons
    area_df: DataFrame containing brain area of all neurons in functional database, uniquely identifiable
    by their (session, scan_idx, unit_id) tuples, or str with the path to the csv file containing the area_df

    Returns:
    v1l234_neur: pd.DataFrame only containing neurons from L2/3/4 of V1

    '''
    funct_match = load_table(client, table_name)
    funct_match_clean = funct_match[['pt_root_id', 'id', 'session', 'scan_idx', 'unit_id', 'pt_position']]
    
    if type(area_df) == str:
        v1_area = pd.read_csv(area_df)
    else:
        v1_area = area_df

    v1_area_subset = v1_area[v1_area['brain_area'] == 'V1']
    v1_neurons = funct_match_clean.merge(v1_area_subset, on = ['session', 'scan_idx', 'unit_id'], how = 'inner')

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
    pre_id: str, column name in connectome df with the ids of the presynaptic neurons on which to match the features df on 
    post_id: str,column name in connectome df with the ids of the postsynaptic neurons on which to match the features df on  
    neuron_id: str, column name in the neurone_features df identifying neuron on which to match on in the connectome
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


def proofread_neurons(proofread_neur, dendrites = False, axons = False):
    '''
    Identify and extract  proofread neurons
    Args:
    proofread_neur: df, cave table with proofreading information
    dendrites: bool, whether to extract neurons with fully proofread dendrites
    axons: bool, whether to extract neurons with fully proofread axons
    
    Returns:
    proofread_neur: DF with information on proofread neurons
    '''

    if dendrites:
        proofread_neur = proofread_neur[(proofread_neur['status_dendrite'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])]
    
    elif axons:
        proofread_neur = proofread_neur[(proofread_neur['status_axon'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])]

    else:
        proofread_neur = proofread_neur[(proofread_neur['status_dendrite'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])&
                                (proofread_neur['status_axon'] ==  'extended')]
    return proofread_neur

# Define the function to determine the new column values
def identify_proofreading_status(df, proofreading_df, id_col = 'pt_root_id'):
    ''' Function to identify the proofreading status of a neuron based on the proofreading table
    Args:
    df: pd.DataFrame, dataframe containing the neurons for which to identify the proofreading status
    proofreading_df: pd.DataFrame, dataframe containing the proofreading information
    id_col: str, name of the column containing the ids of the neurons in the df
    
    Returns:
    str, with the proofreading status of the neuron in the df'''

    #Identify ids of neurons with differing proofreading statuses
    full = set(proofread_neurons(proofreading_df)['pt_root_id'].values)
    dendrites = set(proofread_neurons(proofreading_df, dendrites = True)['pt_root_id'].values)
    axons = set(proofread_neurons(proofreading_df, axons = True)['pt_root_id'].values)
    dendrites_only = dendrites.difference(full)
    axons_only = axons.difference(full)

    
    if df[id_col] in full:
        return 'full'
    elif df[id_col] in full in dendrites_only:
        return 'dendrite'
    elif df[id_col] in full in axons_only:
        return 'axon'
    else:
        return 'not_proofread'
    
if __name__ == '__main__':
    import os
    print(os.getcwd())