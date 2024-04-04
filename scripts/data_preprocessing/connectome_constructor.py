'''
By running this script you can extract a csv file containing a subset of the connectome
with the connnectivity amongst all functionally matched neurons from L2/3/4 of V1 and all other neurons of interest (such as inhibitory neurons)
contained in the unit table.
For each connection you also have the size of the synapse and the difference in preferred orientation between the pre and post synaptic neurons

Estimated runtime: 5 minutes
'''

#TODO
#add argparser option to specify where to read and write files to and which version of the Caveclient database to use

import numpy as np
import pandas as pd 
from ccmodels.preprocessing.connectomics import client_version, connectome_constructor, connectome_feature_merger


def main():
    #define Caveclient and database version
    client = client_version(343)
    
    #Load unit table containing information on desired neurons
    neurons = pd.read_csv('../../data/preprocessed/unit_table.csv')

    #Extracting all the root id of the desired neurons
    neuron_ids = np.array(list(set(neurons[neurons['root_id'] != 0]['root_id'])))

    #Extract connectome
    connections = connectome_constructor(client, neuron_ids, neuron_ids, 500)

    
    #Add information on difference in preferred angle between pre and post synaptic neurons
    connections = connectome_feature_merger(connections, neurons[['root_id','pref_ori']], neuron_id='root_id')
    connections['dtheta'] = connections['post_pref_ori']-connections['pre_pref_ori']

    #Clean up the dataframe by removing unnecessary columns and renaming the size column
    connections_clean = connections.copy()
    connections_clean = connections_clean.drop(columns = ['pre_pref_ori', 'post_pref_ori'])
    connections_clean = connections_clean.rename(columns={'size':'synapse_size'})


    #Save it
    print('Saving connections table')
    connections_clean.to_csv('../../data/preprocessed/connections_table.csv', index = False)
    print('Data Saved')
    
if __name__ == '__main__':
    main()

