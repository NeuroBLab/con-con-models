from caveclient import CAVEclient
import pandas as pd
import numpy as np
import time as time
import requests

def get_client(version = 343):
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


def download_tables(client):

    #--- Functional matched neurons
    print("Download functionally matched neurons...")
    columns_2_download = ['id','pt_root_id', 'session','scan_idx','unit_id', 'pt_position']
    table = load_table(client, 'coregistration_manual_v3', columns=columns_2_download)
    #Drop unlabelled neurons
    table = table[table['pt_root_id']!=0]
    #Drop neurons recorded in more than one scan
    #table = table.drop_duplicates(subset='pt_root_id', keep = 'first')
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



def connectome_constructor(client, presynaptic_set, postsynaptic_set, neurs_per_steps = 500, start_index=0, max_retries=10, delay=5, drop_synapses_duplicates=True):
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
    
    #We are doing the neurons in packages of neurs_per_steps. If neurs_per_steps is not
    #a divisor of the postsynaptic_set the last iteration has less neurons 
    n_before_last = (postsynaptic_set.size//neurs_per_steps)*neurs_per_steps
    
    #Time before starting the party
    time_0 = time.time() 

    #Preset the dictionary so we do not build a large object every time
    syndfs = []
    neurons_to_download = {"pre_pt_root_id":presynaptic_set}
    part = start_index
    for i in range(start_index*neurs_per_steps, postsynaptic_set.size, neurs_per_steps):
        #Inform about our progress
        print(f"Postsynaptic neurons queried so far: {i}...")

        cols_2_download = ["pre_pt_root_id", "post_pt_root_id", "size"]
        #Try to query the API several times
        for retry in range(max_retries):
            try:
                #Get the postids that we will be grabbing in this query. We will get neurs_per_step of them
                post_ids = postsynaptic_set[i:i+neurs_per_steps] if i < n_before_last else postsynaptic_set[i:]
                neurons_to_download["post_pt_root_id"] = post_ids

                #Query the table 
                sub_syn_df = client.materialize.query_table('synapses_pni_2',
                                                    filter_in_dict=neurons_to_download,
                                                    select_columns=cols_2_download)
                


                #Sum all repeated synapses. The last reset_index is because groupby would otherwise create a 
                #multiindex dataframe and we want to have pre_root and post_root as columns
                if drop_synapses_duplicates:
                    sub_syn_df = sub_syn_df.groupby(["pre_pt_root_id", "post_pt_root_id"]).sum().reset_index()

                sub_syn_df.to_csv(f'data/in_processing/connections_table_{part}.csv', index = False)
                part += 1

                #Add the result to the table
                #syndfs.append(sub_syn_df.values)

                #Measure how much time in total our program did run
                elapsed_time = time.time() - time_0

                #Use it to give the user an estimation of the end time.
                neurons_done = i+neurs_per_steps
                time_per_neuron = elapsed_time / neurons_done  
                neurons_2_do = postsynaptic_set.size - neurons_done
                remaining_time = time_format(neurons_2_do * time_per_neuron) 
                print(f"Estimated remaining time: {remaining_time}")
                break
            #If it a problem of the client, just retry again after a few seconds
            except requests.HTTPError as excep: 
                print(f"API error. Retry in {delay} seconds...")
                print(excep)
                time.sleep(delay)
                print(f"Trial {retry} failed. Resuming operations...")
                continue
            #If not, just raise the exception and that's all
            except Exception as excep: 
                raise excep

        #If the above loop did not succeed for any reason, then just abort.
        if retry >= max_retries:
            raise TimeoutError("Exceeded the max_tries when trying to get synaptic connectivity")
    
    #syn_df = pd.DataFrame({'pre_id':np.vstack(syndfs)[:, 0], 'post_id': np.vstack(syndfs)[:, 1], 'syn_volume': np.vstack(syndfs)[:, 2]})
    #return syn_df
    return


def time_format(seconds):
    if seconds > 3600*24: 
        days = int(seconds//(24*3600))
        hours = int((seconds - days*24*3600)//3600)
        return f"{days} days, {hours}h"
    elif seconds > 3600: 
        hours = int(seconds//3600)
        minutes = int((seconds - hours*3600) // 60)
        return f"{hours}h, {minutes}min"
    elif seconds > 60:
        minutes = int(seconds//60)
        rem_sec = int((seconds - 60*minutes))
        return f"{minutes}min {rem_sec}s"
    else:
        return f"{seconds:.0f}s"