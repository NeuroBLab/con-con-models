'''
This script allows to extract and save a connectivity sample fro proofread and for non-proofread neurons
on which to compute relevant statistics and required analyses.
'''

import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from caveclient import CAVEclient
from standard_transform import minnie_transform_vx
from ccmodels.analysis.utils import layer_extractor, unique_neuronal_inputs, unique_neuronal_outputs

def client_version(version = 'minnie65_public_v343'):
    '''Define the version of the CAVE client database you want to access and use for the analysis'''
    client = CAVEclient(version)

    return client


def neuron_identifier(client, table):
    '''Extract the root_ids of the cells that are nuerons
    Args:
    client: CAVEclient
    table: str, name of cave table to query
    
    Returns: 
    neurons: dict, with root_ids of neurons
    cell_table: DF, with information on all cells in the dataset
    '''

    # Set of cells in the connectome
    cell_table = client.materialize.query_table(table)
    #select all the ids of cells that are actually  neurons
    neurons = set(cell_table[cell_table['cell_type'] == 'neuron']['pt_root_id'].values)

    return neurons, cell_table


def proofread_neurons(client, table):
    '''
    Identify and extract fully proofread neurons
    Args:
    client: CAVEclient
    table: str, name of cave table to query
    
    Returns:
    proofread_neur: DF with information on proofread neurons
    '''

    # Set of fully proofread neurons
    proofread_neur = client.materialize.query_table(table)
    proofread_neur = proofread_neur[(proofread_neur['status_dendrite'] == 'extended') & 
                                (proofread_neur['pt_root_id'] == proofread_neur['valid_id'])&
                                (proofread_neur['status_axon'] ==  'extended')]
    return proofread_neur

def nonproofread_neurons(neurons, proofread_neur, cell_table):
    '''
    Identify and extract non proofread neurons
    Args:
    neurons: dict, with root_ids of neurons
    proofread_neur: DF with information on proofread neurons
    cell_table: DF, with information on all cells in the dataset

    Returns:
    noproof_neur: DF with information on nonproofread neurons
    '''
    
    #Ids of proofread neurons
    proofread_ids = set(proofread_neur['pt_root_id'])

    # Set of NON proofread neurons
    noproof_ids = neurons.difference(proofread_ids)
    noproof_neur = cell_table[(cell_table['pt_root_id'].isin(noproof_ids)) & (cell_table['pt_root_id']!=0) ]

    return noproof_neur


def sampler(client, neurons, neurons_subtype,  n = 100, seed = 4, extract = 'both', name = 'Proofread'):
    '''Reconstruct the presynaptic and postsynaptic connections of a sample of proofread or nonproofread neurons'''

    #Define seed and transformation
    np.random.seed(seed)

    tform_vx = minnie_transform_vx()

    #Iterating through subset of proofread/nonproofread neurons
    neur_out_t = unique_neuronal_outputs(neurons_subtype['pt_root_id'].values[0], neurons, client)
    neur_in_t = unique_neuronal_inputs(neurons_subtype['pt_root_id'].values[0], neurons, client)

    for cell in tqdm(np.random.choice(neurons_subtype['pt_root_id'].values[1:], n, replace = False), desc = f'Extracting sample for {name} neurons'):
        if extract == 'both':
            #Concatenate pre/post information on extracted neuron to shared DataFrame
            neur_out = unique_neuronal_outputs(cell, neurons, client)
            neur_out_t = pd.concat([neur_out_t, neur_out], axis = 0)

            neur_in = unique_neuronal_inputs(cell, neurons, client)
            neur_in_t = pd.concat([neur_in_t, neur_in], axis = 0)

        elif extract == 'outputs':
            neur_out = unique_neuronal_outputs(cell, neurons, client)
            neur_out_t = pd.concat([neur_out_t, neur_out], axis = 0)
        
        elif extract == 'inputs':
            neur_in = unique_neuronal_inputs(cell, neurons, client)
            neur_in_t = pd.concat([neur_in_t, neur_in], axis = 0)

        else:
            raise Exception("You need to specifiy a valid type of data to extract, accepted values are: 'inputs', 'outputs' or 'both'.")

    if extract == 'both':
        #Extract cortical layers of pre andpost synaptic neurons
        neur_out_connections = layer_extractor(neur_out_t, tform_vx)
        neur_in_connections = layer_extractor(neur_in_t, tform_vx)

        return neur_out_connections, neur_in_connections
    
    elif extract == 'outputs':
        neur_out_connections = layer_extractor(neur_out_t, tform_vx)
        return neur_out_connections
    
    elif extract == 'inputs':
        neur_in_connections = layer_extractor(neur_in_t, tform_vx)
        return neur_in_connections



def tuning_identifier(neur_in_connections, v1neurons_path = '../con-con-models/data/v1l234_neurons.pkl'):
    '''Identifies if the sample of presynaptic neurons are tuned or not
    by looking at whether they are neurons from the tuning analysis present in the pre-processing'''

    v1_neurons = pd.read_pickle(v1neurons_path)
    tuned_neurons = v1_neurons[v1_neurons['model_type']!= 'not_selective']

    #Assign whether pre neurons are tuned or untuned in proofread neurons
    tuned_untuned = []
    for i in neur_in_connections['pre_pt_root_id']:
        if i in v1_neurons['root_id'].values:
            if i in set(tuned_neurons['root_id']):
                tuned_untuned.append('tuned')
            else:
                tuned_untuned.append('untuned')
        else:
            tuned_untuned.append('nan')

    neur_in_connections['is_pre_tuned'] = tuned_untuned

    return neur_in_connections

def csv_samp_saver(file, save_path, file_name):
    '''Save the desired file in csv format at the desired location and with the specified filename'''
    file.to_csv(f'{save_path}/{file_name}.csv')


def main():
    '''Main function collecting the sample of proofread and non-proofread neurons for analysis'''
    
    #Argparser to specify where to save file on machine
    parser = argparse.ArgumentParser(description='''Generate the pre and post synaptic connections for a sample of 
                                     Proofread and Nonproofread neurons''')
    
    # Adding and parsing arguments
    parser.add_argument('destination', type=str, help='Destination path to save sample in')
    args = parser.parse_args()

    print('''
          
    Setting up sample extraction requirements...
          
          ''')
    #set up client with database version
    client = client_version()

    #Extract information on which cells are neurons
    neurons, cell_table = neuron_identifier(client, 'nucleus_neuron_svm')

    #Extract proofread neurons
    proofread_neur = proofread_neurons(client, 'proofreading_status_public_release')

    #Extract nonproofread neurons
    noproof_neur = nonproofread_neurons(neurons, proofread_neur, cell_table)

    #print(noproof_neur.head())
    #Add layer information to the neurons
    tform_vx = minnie_transform_vx()
    proofread_neur = layer_extractor(proofread_neur, tform_vx, column='pt_position')
    noproof_neur = layer_extractor(noproof_neur, tform_vx, column='pt_position')

    print('''
          
    Starting sample extraction
          
          ''')
    #Extract connectome sample of proofread neurons
    proof_out_connections, proof_in_connections = sampler(client, neurons, proofread_neur,  n = 100, 
                                                          seed = 4, extract = 'both', name = 'Proofread')
    
    #Extract connectome sample of nonproofread neurons
    noproof_out_connections, noproof_in_connections = sampler(client, neurons, noproof_neur,  n = 100, 
                                                              seed = 4, extract = 'both', name = 'Nonproofread')

    #Exract information on proportions of tuned and untuned neurons
    proof_in_connections = tuning_identifier(proof_in_connections)

    noproof_in_connections = tuning_identifier(noproof_in_connections)

    #Identify only neurons form L2/3
    proofread_neurl23 = proofread_neur[proofread_neur['cortex_layer'] == 'L2/3']
    noproof_neurl23 = noproof_neur[noproof_neur['cortex_layer'] == 'L2/3']

    proof_in_connectionsl23 = sampler(client, neurons, proofread_neurl23,  n = 100, 
                                                          seed = 4, extract = 'inputs', name = 'Proofread L2/3')
    
    noproof_in_connectionsl23 = sampler(client, neurons, noproof_neurl23,  n = 100, 
                                                          seed = 4, extract = 'inputs', name = 'Nonproofread L2/3')

    print('''
          
    Extraction finished saving data
          
          ''')

    #Save the samples in the desired location
    csv_samp_saver(proof_in_connections, args.destination, 'proofread_inputs_sample')
    csv_samp_saver(proof_out_connections, args.destination, 'proofread_outputs_sample')
    csv_samp_saver(noproof_in_connections, args.destination, 'nonproof_inputs_sample')
    csv_samp_saver(noproof_out_connections, args.destination, 'nonproof_outputs_sample')
    csv_samp_saver(proof_in_connectionsl23, args.destination, 'proof_inputs_l23sample')
    csv_samp_saver(noproof_in_connectionsl23, args.destination, 'nonproof_inputs_l23sample')

    print('''
          
    Data Saved Succesfully!
          
          ''')

if __name__ == '__main__':
    main()