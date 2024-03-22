''''This script is used to assign the selectivity type, layer, OSI, proofreading status, position, cell type 
to functionally matched L2/3/4 V1 neurons and inhibitory L2/3 V1 neurons.'''

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ccmodels.preprocessing.utils import tuning_labler, min_act, osi_calculator, angle_indexer
from ccmodels.preprocessing.connectomics import subset_v1l234, client_version,identify_proofreading_status



#Read in data with selectivity infromation
l234_orifits = pd.read_pickle('../../data/in_processing/orientation_fits.pkl')

# Identify the tuning types of neurons based on pvalue and r squared thresholds
neur_seltype = tuning_labler(l234_orifits)

# Subset only the neurons in L2/3 and 4 of V1 to get layer, area and position labels
client = client_version(661)
func_neurons = subset_v1l234(client, area_df = '../../data_full/v1_n.csv')
#split column with position values in to three separate columns
func_neurons[['x_pos', 'y_pos', 'z_pos']] = func_neurons['pial_distances'].apply(lambda x: pd.Series(x))


#Merge to get information on layer
neur_seltype = neur_seltype.merge(func_neurons, left_on=['root_id', 'session', 'scan_idx', 'unit_id'], right_on = ['pt_root_id', 'session', 'scan_idx', 'unit_id'], how = 'inner')


# Calculate the OSI of neurons
#Directions shown during stimulus presentation
arr_dirs = np.array(sorted(list(set(neur_seltype['orientations'].values[0]))))

osis = []
for pref_ori, tunig_type, responses in zip(neur_seltype['phi'].values, neur_seltype['tuning_type'].values, neur_seltype['activity']):
    
    #Find the least preferred orientation depending on the model type
    least_pref_ori = min_act(pref_ori, tunig_type, arr_dirs)
    #Calculate the OSI
    osi = osi_calculator(least_pref_ori, pref_ori, responses, arr_dirs)
    osis.append(osi)

#Assign new osi column
neur_seltype['osi'] = osis

#Add indexed version of the preferred angle
neur_seltype['pref_ori'] = neur_seltype['phi'].apply(angle_indexer, axis = 1)

# Add inhibitory neurons and relevant column to identify them
inhib_neurons = pd.read_pickle('/Users/jacopobiggiogera/Desktop/con-con-models/data/in_processing/inhibitory_nurons_ba.pkl')
nuc = client.materialize.query_table('aibs_soma_nuc_metamodel_preds_v117')

#Add layer information
inhib_neurons_l = layer_extractor(inhib_neurons, tform_vx, column='pt_position')

# Identify the ones in v1 L2/3
inhv1l23 = inhib_neurons_l[(inhib_neurons_l['brain_area'] == 'V1') & (inhib_neurons_l['cortex_layer'] == 'L2/3')]
inhv1l23[['x_pos', 'y_pos', 'z_pos']] = inhv1l23['pial_distances'].apply(lambda x: pd.Series(x))

neurs = nuc[nuc['classification_system']=='aibs_neuronal']
inhib_neur = inhv1l23.merge(neurs, on='id', how='inner')

#Add proofreading information
neur_seltype['proofreading'] = neur_seltype.apply(identify_proofreading_status, axis=1)