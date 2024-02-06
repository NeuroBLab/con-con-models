'''
By running this script you can extract a pickle file containing a subset of teh connectome
with the connnectivity amongst all functionally matched neurons from L2/3/4 of V1.
In addition pre and post synpatic neurons also contain information on...

Estimated runtime: 1 hour
'''

#TODO
#add argparser option to specify where to read and write files to

import numpy as np
import pandas as pd 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from caveclient import CAVEclient
from ccmodels.preprocessing.extractors.utils import connectome_constructor, subset_v1l234, constrain_act_range, constrainer, connectome_feature_merger


def main():
    #define Caveclient and database version
    client = CAVEclient('minnie65_public')
    client.materialize.version = 661

    ############################### Exctract connectome of functionally matched L2/3/4 V1 neurons ##################################
    v1l234_neur = subset_v1l234(client, table_name = 'coregistration_manual_v3', area_df = 'con-con-models/data_full/v1_n.csv')

    #Extracting all the root id of the functionally matched cells
    nv1l234 = np.array(list(set(v1l234_neur[v1l234_neur['pt_root_id'] != 0]['pt_root_id'])))

    #Extract connectome
    connect_v1l234 = connectome_constructor(client, nv1l234, nv1l234, 500)

    
    ############################### Merge connectome DF with DF with information on neuron selectivity ##################################
    v1l234_neurons = pd.read_pickle('/Users/jacopobiggiogera/Desktop/con-con-models/data/v1l234_neurons.pkl')
    connect_v1l234 = connectome_feature_merger(connect_v1l234, v1l234_neurons, neuron_id='root_id')

    ############################### Sum the synaptic strength of neuron pairs with repeated connections ##################################
    #Drop repeated connections
    v1l234_connections_nodup = connect_v1l234.drop_duplicates(subset = ['post_pt_root_id', 'pre_pt_root_id'])

    #Sum synaptic strength of repeated connections
    summed_synapses = connect_v1l234.groupby(['post_pt_root_id', 'pre_pt_root_id'])['size'].sum().reset_index()

    #Merge the new summed synapses
    v1l234_connections_nodup = v1l234_connections_nodup.merge(summed_synapses, on = ['post_pt_root_id', 'pre_pt_root_id'], how = 'inner')

    #Rename new connection strength and remove old one
    v1l234_connections_clean = v1l234_connections_nodup.rename(columns= {'size_y':'size'})
    v1l234_connections_clean = v1l234_connections_clean.drop(columns = 'size_x')
    
    ############################### Add current information and information on preferred orientation ##############################
    # Calculate the presynaptic current
    v1l234_connections_clean['current'] = v1l234_connections_clean['pre_activity'].values * v1l234_connections_clean['size'].values

    #Calculate difference between preferred pre synaptic and preferred post synaptic orientation
    v1l234_connections_clean['delta_ori'] = v1l234_connections_clean['pre_po']- v1l234_connections_clean['post_po']

    
    ############################### Constrain directions in the range [-pi, pi] ########################################
    v1l234_connections_clean = v1l234_connections_clean.sort_values('post_pt_root_id')

    #discretized direction
    arr_dirs = np.array(sorted(list(set(v1l234_connections_clean['pre_po'].values))))

    #Select set of post synaptic cells
    cells = sorted(set(v1l234_connections_clean['post_pt_root_id']))

    #Initiate containers for new directions and activities
    re_curs = []
    re_acts = []
    dirs = []
    #Iterate cells and extract new directions and reordered activities
    for i in tqdm(cells, desc= 'Constraining oreintations in (-pi, pi]'):
        #Reorder currents
        ra, dr = constrain_act_range('post_pt_root_id', i, arr_dirs, v1l234_connections_clean)
        dirs += [dr]*len(ra)
        re_curs+=ra

        #Reorder activities
        acts, _ = constrain_act_range('post_pt_root_id', i, arr_dirs, v1l234_connections_clean, currents = False)
        re_acts+=acts

    #assign them to new columns
    v1l234_connections_clean['shifted_current'] = re_curs
    v1l234_connections_clean['shifted_activity'] = re_acts
    v1l234_connections_clean['new_dirs'] = dirs

    #Generating constrained column for maxpre-maxpost
    v1l234_connections_clean['delta_ori_constrained'] = constrainer(v1l234_connections_clean['delta_ori'].values)
    v1l234_connections_clean['delta_ori_constrained'] = v1l234_connections_clean['delta_ori_constrained'].round(6)
    v1l234_connections_clean['delta_ori_constrained'] = v1l234_connections_clean['delta_ori_constrained'].replace(-3.141593, 3.141593)

    #Save it
    v1l234_connections_clean.to_pickle('v1l234_connections.pkl')

if __name__ == '__main__':
    main()

