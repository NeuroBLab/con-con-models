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

def load_table(client, name, columns=[]):
    '''Function to load a table from the CAVEclient database
    
    Args:
    client: CAVEclient 
        Needed to access MICrONS connectomics data
    name: str 
        Name of the table to load
    columns: list of str 
        Name of the columns to include. If empty (default) all columns are returned
    
    Returns:
    df: pd.DataFrame with the table loaded from the database'''

    if len(columns)>0: 
        return client.materialize.query_table(name, select_columns=columns, split_positions=True)
    else:
        return client.materialize.query_table(name, split_positions=True)


def read_table(name, prepath="data/", splitpos=False):
    """
    Read an already stored raw/in-processing data table. 
    For loading the preprocessed data, please refer to the datanalysis module.
    """

    #Get the path to the table and read it
    path = prepath + _get_table_path(name)

    if "csv" in path:
        table = pd.read_csv(path)
    elif "pkl" in path:
        table = pd.read_pickle(path) 

    #Merge together the positions, so the column contains a list with 3 values
    if "pt_position_x" in table.columns and not splitpos:
        splitpos = ["pt_position_x", "pt_position_y", "pt_position_z"]
        table["pt_position"] = list(table[splitpos].values)  
        table.drop(columns=splitpos, inplace=True)

    #Return the thing
    return table

def _get_table_path(name):
    """
    Not intended for use outside of the module. 
    Auxiliary function that indexes in which folder each file is, and return its path. 
    """
    folder_content = {"in_processing" : ["ei_table.csv", "orientation_fits.pkl"], 
                      "raw" : ["aibsct.csv", "area_membership.csv", "coarsect.csv", "finect.csv", 
                               "functionally_matched.csv", "neuronref.csv", "proofreading.csv"]}

    #Check if the name is in one of the filenames...
    for key in folder_content:
        files_in_folder = folder_content[key]
        for filename in files_in_folder:
            #If it is, just return the path
            if name in filename: 
                return f"{key}/{filename}"




def download_tables(client):

    #--- Functional matched neurons
    print("Download functionally matched neurons...")
    columns_2_download = ['id','pt_root_id', 'session','scan_idx','unit_id', 'pt_position']
    table = load_table(client, 'coregistration_manual_v3', columns=columns_2_download)
    #Drop unlabelled neurons
    table = table[table['pt_root_id']!=0]
    #Drop neurons recorded in more than one scan
    table = table.drop_duplicates(subset='pt_root_id', keep = 'first')
    #Save the table
    table.to_csv("data/raw/functionally_matched.csv", index=False)

    #--- Are observed nuclei a neuron? 
    print("Download nucleus classification reference...")
    columns_2_download = ['id', 'target_id', 'pt_root_id','cell_type','pt_position']
    table = load_table(client, 'nucleus_ref_neuron_svm', columns=columns_2_download)
    table = table[table['pt_root_id'] !=0]
    #table = table.drop_duplicates(subset='pt_root_id', keep = 'first')
    table.to_csv("data/raw/neuronref.csv", index=False)

    # --- Excitatory or inhibitory?
    print("Download Baylor et al coarse cell classification...")
    table = load_table(client, 'baylor_log_reg_cell_type_coarse_v1', columns=columns_2_download)
    table = table[table['pt_root_id'] != 0]
    #Some tables have some duplicates. I manually checked that all are the same objects,
    #by checking they have the same positions. I do not want those duplicates 
    table.drop_duplicates(subset = "target_id", inplace=True)
    table.to_csv("data/raw/coarsect.csv", index=False)

    # --- Fine-type classification
    print("Download Baylor et all fine cell classification...")
    table = load_table(client, 'baylor_gnn_cell_type_fine_model_v2', columns=columns_2_download)
    table = table[table['pt_root_id'] !=0]
    table.drop_duplicates(subset = "target_id", inplace=True)
    table.to_csv("data/raw/finect.csv", index=False)

    # --- AIBS classification  
    print("Download AIBS classification...")
    columns_2_download = ['id', 'target_id', 'pt_root_id', 'classification_system', 'cell_type', 'pt_position']
    table = load_table(client, 'aibs_soma_nuc_metamodel_preds_v117', columns=columns_2_download)
    table = table[table['pt_root_id'] !=0]
    table.drop_duplicates(subset = "target_id", inplace=True)
    table.to_csv("data/raw/aibsct.csv", index=False)

    # --- Proofreading status 
    print("Download proofreading status...")
    columns_2_download = ['pt_root_id', 'valid_id', 'status_dendrite', 'status_axon', 'pt_position']
    table = load_table(client, 'proofreading_status_public_release', columns=columns_2_download)
    table = table[table['pt_root_id'] !=0]
    #For this table in particular we have that these are the only valid ids. After that, we do not need that column
    table = table[table['pt_root_id'] == table['valid_id']]
    table = table[['pt_root_id', 'status_dendrite', 'status_axon', 'pt_position_x', 'pt_position_y', 'pt_position_z']] 
    table.to_csv("data/raw/proofreading.csv", index=False)

    print("Download completed succesfully.")



def obtain_ei_table(prepath="data/"):
    """
    This function combines all the information avaiable from three tables in the Microns project to get a slightly 
    better estimation of an object being a neuron and whether if it's excitatory or inhibitory.

    It prodcues a new table with the classified IDs in the 'in_processing' folder.
    """

    #Load all the tables
    neuronref   = read_table("neuronref", prepath=prepath)
    coarsedata  = read_table("coarsect", prepath=prepath)
    finedata    = read_table("finect", prepath=prepath)
    aibs        = read_table("aibsct", prepath=prepath)

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


def subset_v1l234(path="data/"):
    '''This function takes a table of functionally matched neurons from the MICrONs connectomics database
    and returns a subset only containing neurons belonging to L2/3/4 of V1
    
    Returns:
    v1l234_neur: pd.DataFrame only containing neurons from L2/3/4 of V1
    '''

    #Get the functionally matched neurons
    #funct_match = pd.read_csv(f"{path}/functionally_matched.csv")
    funct_match = read_table("functionally_matched", prepath=path)
    funct_match_clean = funct_match[['pt_root_id', 'id', 'session', 'scan_idx', 'unit_id', 'pt_position']]
    
    #Get the neurons that live in V1
    #v1_area = pd.read_csv(f"{path}/area_membership.csv")
    v1_area = read_table("area_membership", prepath=path)
    v1_area_subset = v1_area[v1_area['brain_area'] == 'V1']

    #Merge tables to get only functionally matched neurons in V1
    v1_neurons = funct_match_clean.merge(v1_area_subset, on = ['session', 'scan_idx', 'unit_id'], how = 'inner')

    #Get the layer of such neurons
    tform_vx = minnie_transform_vx()
    v1_neurons_layers = layer_extractor(v1_neurons, tform_vx, column = 'pt_position')

    #Finally return the ones in L2/3 or L4 
    return v1_neurons_layers[v1_neurons_layers['layer'].isin(['L2/3', 'L4'])]

    

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


    
if __name__ == '__main__':
    import os
    print(os.getcwd())