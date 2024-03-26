''''This script is used to assign the selectivity type, layer, OSI, proofreading status, position, cell type 
to functionally matched L2/3/4 V1 neurons and inhibitory L2/3 V1 neurons.'''

import numpy as np
import pandas as pd
from ccmodels.preprocessing.utils import tuning_labler, min_act, osi_calculator, angle_indexer, layer_extractor
from ccmodels.preprocessing.connectomics import subset_v1l234, client_version,identify_proofreading_status, load_table
from standard_transform import minnie_transform_vx

tform_vx = minnie_transform_vx()

#Read in data with selectivity infromation
l234_orifits = pd.read_pickle('../../data/in_processing/orientation_fits.pkl')

print('Assigning selectivity type to neurons')
# Identify the tuning types of neurons based on pvalue and r squared thresholds
neur_seltype = tuning_labler(l234_orifits)

# Subset only the neurons in L2/3 and 4 of V1 to get layer, area and position labels
client = client_version(343) #select the CaveDatabase version you are interested in 
func_neurons = subset_v1l234(client,table_name='functional_coreg', area_df = '../../data/raw/area_membership.csv')
#split column with position values in to three separate columns
func_neurons[['x_pos', 'y_pos', 'z_pos']] = func_neurons['pial_distances'].apply(lambda x: pd.Series(x))

#Merge to get information on layer
neur_seltype = neur_seltype.merge(func_neurons, left_on=['root_id', 'session', 'scan_idx', 'cell_id'], right_on = ['pt_root_id', 'session', 'scan_idx', 'unit_id'], how = 'inner')


# Calculate the OSI of neurons
print('Calculating OSI for each neuron')
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
neur_seltype['pref_ori'] = neur_seltype['phi'].apply(angle_indexer)

# Add inhibitory neurons and relevant column to identify them
print('Adding information on inhibitory neurons')
inhib_neurons = pd.read_pickle('../../data/in_processing/inhibitory_neurons_bal.pkl')

# Identify the ones in v1 L2/3
inhv1l23 = inhib_neurons[(inhib_neurons['brain_area'] == 'V1') & (inhib_neurons['cortex_layer'] == 'L2/3')]
inhv1l23[['x_pos', 'y_pos', 'z_pos']] = inhv1l23['pial_distances'].apply(lambda x: pd.Series(x))

#Add columns to match the excitatory neurons df
inhv1l23['tuning_type'] = np.nan*len(inhv1l23)
inhv1l23['osi'] = np.nan*len(inhv1l23)
inhv1l23['pref_ori'] = np.nan*len(inhv1l23)
inhv1l23['root_id'] = inhv1l23['pt_root_id']

neur_seltype['cell_type'] = 'excitatory'

#Select only the relevant columns
inhib_clean = inhv1l23[['root_id','pref_ori', 'cell_type', 'tuning_type','osi', 'brain_area', 'cortex_layer', 'x_pos',
       'y_pos', 'z_pos']]

excit_clean = neur_seltype[['root_id','pref_ori','cell_type',  'tuning_type','osi', 'brain_area', 'cortex_layer', 'x_pos',
       'y_pos', 'z_pos']]

#Concatenate the inhibitory and excitatory neuron dataframes
neurs_all = pd.concat([excit_clean, inhib_clean], axis = 0)

#Add proofreading information
print('Adding proofreading information')
proofread = load_table(client, 'proofreading_status_public_release')
neurs_all['proofreading'] = neurs_all.apply(identify_proofreading_status, axis=1, args = (proofread,'root_id' ))

#Save the dataframe
print('Saving the data')
neurs_all.to_csv('../../data/preprocessed/unit_table.csv', index = False)
print('Data saved') 