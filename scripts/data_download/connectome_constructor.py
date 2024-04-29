'''
By running this script you can extract a csv file containing a subset of the connectome
with the connnectivity amongst all functionally matched neurons from L2/3/4 of V1 and all other neurons of interest (such as inhibitory neurons)
contained in the unit table.
For each connection you also have the size of the synapse and the difference in preferred orientation between the pre and post synaptic neurons

Estimated runtime: 5 minutes
'''

#TODO
#add argparser option to specify where to read and write files to and which version of the Caveclient database to use

import sys
sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
import numpy as np
import pandas as pd 
import ccmodels.preprocessing.connectomics as conn

#define Caveclient and database version
client = conn.client_version(661)

#Load unit table containing information on desired neurons
neurons = pd.read_csv('data/preprocessed/unit_table.csv')

#Extracting all the root id of the desired neurons
neuron_ids = neurons["pt_root_id"].unique()

#Extract connectome
print("Querying API for connections. Might take a while...")
connections = conn.connectome_constructor(client, neuron_ids, neuron_ids)

#Save it
connections.to_csv('data/preprocessed/connections_table.csv', index = False)
print('Data Saved')

