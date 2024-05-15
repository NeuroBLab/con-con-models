import numpy as np
import pandas as pd
from standard_transform import minnie_transform_vx
import ccmodels.preprocessing.rawloader as loader
import os


def layer_extractor(input_df, transform, column = 'pre_pt_position'):
    '''This function assigns a layer to each neuron based on the y axis value of the pial distance
    
    Args:
    input_df: pandas dataframe containing the 3d coordinates
    transform: transform object to turn the 3d coordinates in to pial distances 
    column: string, column name containing the pial distances
    
    Returns:
    input_df: pandas dataframe containing the pial distances and the assigned layer
    '''
    input_df['pial_distances'] = transform.apply(input_df[column])

    #Use the y axis value to assign the corresponding layer as per Ding et al. 2023
    layers = []
    for i in input_df['pial_distances'].iloc[:]:
        if 0<i[1]<=98:
            layers.append('L1')
        elif 98<i[1]<=283:
            layers.append('L23')
        elif 283<i[1]<=371:
            layers.append('L4')
        elif 371<i[1]<=574:
            layers.append('L5')
        elif 574<i[1]<=713:
            layers.append('L6')
        else:
            layers.append('unidentified')

    input_df['layer'] = layers   
    return input_df

def obtain_ei_table(prepath="data/"):
    """
    This function combines all the information avaiable from three tables in the Microns project to get a slightly 
    better estimation of an object being a neuron and whether if it's excitatory or inhibitory.

    It prodcues a new table with the classified IDs in the 'in_processing' folder.
    """

    #Load all the tables
    neuronref   = loader.read_table("neuronref", prepath=prepath)
    coarsedata  = loader.read_table("coarsect", prepath=prepath)
    finedata    = loader.read_table("finect", prepath=prepath)
    aibs        = loader.read_table("aibsct", prepath=prepath)

    #From aibs/neuronref, select only the ones that are neurons
    aibs_neurons = aibs[aibs["classification_system"] == "aibs_neuronal"]
    isneuron = neuronref[neuronref["cell_type"] == "neuron"]

    #Which neuron types are excitatory or inhibitory
    types_E = ["23P", "4P", "6P-IT", "6P-CT", "5P-IT", "5P-ET", "5P-PT", "5P-NP"]  
    #types_I = ["MC", "BPC", "NGC", "BC"]

    #Merge all tables with cell type info together, taking only common target_id among them
    checkneuro = coarsedata.merge(finedata,     on=["target_id"], how="inner")
    checkneuro = checkneuro.merge(aibs_neurons, on=["target_id"], how="inner")

    #Find the type of each cell in the merged table
    #cell_type_x is the type in coarse (either exc or inh)
    #cell_type_y and cell_type are the types in finedata and aibs, respectively, and its a string representing a neuron type
    ct = []
    for ct1, ct2, ct3 in checkneuro[["cell_type_x", "cell_type_y", "cell_type"]].values:
        #Number of tables agreeing on the neuron being Exc
        ntype = (ct1 == "excitatory") + (ct2 in types_E) + (ct3 in types_E) 

        #If 2 or more tables agree it's exc 
        if ntype >= 2:
            ct += ["exc"]
        else:
            ct += ["inh"]


    #Copy the indices and create a new column with the inferred cell type
    result = checkneuro[["target_id", "pt_root_id"]].copy()
    result["cell_type"] = ct

    #Merge them with neuronref, to make 100% everything we pick is a neuron 
    #(So we force aibs and neuronref to agree)
    result = result.merge(isneuron, on=["target_id"], how="inner")
    result = result[["target_id", "pt_root_id_x", "cell_type_x", "pt_position"]]
    result = result.rename(columns={"pt_root_id_x":"pt_root_id", "cell_type_x":"cell_type"})

    tform_vx = minnie_transform_vx()
    result = layer_extractor(result, tform_vx, column = 'pt_position')

    return result


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


def get_func_match_subset_v1l234(path="data/"):
    '''This function takes a table of functionally matched neurons from the MICrONs connectomics database
    and returns a subset only containing neurons belonging to L2/3/4 of V1
    
    Returns:
    v1l234_neur: pd.DataFrame only containing neurons from L2/3/4 of V1
    '''

    #Get the functionally matched neurons
    #funct_match = pd.read_csv(f"{path}/functionally_matched.csv")
    funct_match = loader.read_table("functionally_matched", prepath=path)
    funct_match_clean = funct_match[['pt_root_id', 'id', 'session', 'scan_idx', 'unit_id', 'pt_position']]
    
    #Get the neurons that live in V1
    #v1_area = pd.read_csv(f"{path}/area_membership.csv")
    v1_area = loader.read_table("area_membership", prepath=path)
    v1_area_subset = v1_area[v1_area['brain_area'] == 'V1']

    #Merge tables to get only functionally matched neurons in V1
    v1_neurons = funct_match_clean.merge(v1_area_subset, on = ['session', 'scan_idx', 'unit_id'], how = 'inner')

    #Get the layer of such neurons
    tform_vx = minnie_transform_vx()
    v1_neurons_layers = layer_extractor(v1_neurons, tform_vx, column = 'pt_position')

    #Finally return the ones in L2/3 or L4 
    return v1_neurons_layers[v1_neurons_layers['layer'].isin(['L23', 'L4'])]

    

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


def identify_proofreading_status(df, proofreading_df):
    ''' Function to identify the proofreading status of a neuron based on the proofreading table
    Args:
    df: pd.DataFrame, dataframe containing the neurons for which to identify the proofreading status
    proofreading_df: pd.DataFrame, dataframe containing the proofreading information
    
    Returns:
    str, with the proofreading status of the neuron in the df'''

    #Select only the columns we are going to work with to avoid problems
    proofreading_df = proofreading_df[["pt_root_id", "status_axon", "status_dendrite"]]

    #Merge, leaving all the coinciding indices as NaN
    info = df.merge(proofreading_df, on=["pt_root_id"], how="left")

    #Fill the NaN with the default "no proofread" value
    nan_places = info["status_axon"].isna()
    info.loc[nan_places, ["status_axon", "status_dendrite"]] = ["non", "non"]  

    return info 


def merge_connection_tables(prepath="data"):
    #Count the number of tables to merge, by checking all files in the correct folder
    ntables = 0
    for file in os.listdir(f"{prepath}/in_processing"):
        if os.path.isfile(f"{prepath}/in_processing/{file}"):
            if "connections_table_" in file:
                ntables += 1

    #Merge all of them
    table = pd.read_csv(f"{prepath}/in_processing/connections_table_0.csv")
    ntables
    for i in range(1, ntables):
        table = pd.concat([table, pd.read_csv(f"{prepath}/in_processing/connections_table_{i}.csv")])

    
    return table.rename(columns={"size":"syn_volume"})



    
if __name__ == '__main__':
    import os
    print(os.getcwd())