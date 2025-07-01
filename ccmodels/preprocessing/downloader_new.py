import pandas as pd
import numpy as np

import sys
import argparse

import microns_datacleaner as mic

# ----------------------- User input ----------------------------------

parser = argparse.ArgumentParser(description='''Data download and processing''')

# Adding and parsing arguments
parser.add_argument('download_nucleus', type=bool, help='Boolean. If true, downloads all nucleus data')
parser.add_argument('download_synapses', type=bool, help='Boolean. If true, downloads all the synapse data as well')
args = parser.parse_args()

# -------------------- Nucleus download --------------------------

cleaner = mic.MicronsDataCleaner(datadir="data/", version=1300)

#Download all data related to nucleus: reference table, various classifications, functional matchs
if args['download_nucleus']:
    cleaner.download_nucleus_data()

# ------------ Format nucleus table with functional stuff ----------------

#Process the data and obtain units and segment tables
units, segments = cleaner.process_nucleus_data()
units.loc[units['layer']=='L2/3', 'layer'] = 'L23'

#Load our processed functional properties from the coregistration manual v4
funcprops= pd.read_pickle("in_processing/stuff_for_unit_and_activity_table.pkl")
funcprops = funcprops.rename(columns={"pref_ori":"pref_ori_func"})
funcprops['target_id'] = funcprops['target_id'].astype(int)

#Filter the columns we will actually use 
#TODO just do not generate the table like this
funcprops = funcprops.drop(columns=['dirs', 'oris', 'rate_dirs', 'rate_dirs_sem', 'rate_oris', 'rate_oris_sem'])

#Merge to have everything.This table has NaNs on the pref_ori / gOSI of non-matched neurons of coregistration v3 
#and NANs on tuning_type/pref_ori_func of coregistration v4
units_with_func = units.merge(funcprops, left_on='id', right_on='target_id', how='left') 

#These 3 neurons are found in the coregistration table so they should be excitatory, but the AIBS classification say they are nonneuronal.
#Since the coregistration is manual, we favor that one and set the neurons manually to be excitatory so when we filter they do not disappear
unit_table = units_with_func.copy()
unit_table.loc[(unit_table['pt_root_id'].isin([864691135726289983,864691135569616364,864691136085014636])), 'classification_system'] = 'excitatory_neuron'

#Filter for our layer of interest
#TODO use the filters!
unit_table = units_with_func.loc[(units_with_func['brain_area']=='V1')&(units_with_func['layer'].isin(['L23', 'L4']))&(units_with_func['classification_system']!='nonneuron'), :]

#Get the neurons not matched in coreg v4:
not_matched = unit_table['tuning_type'].isna()
#I will eliminate all the nans and just say the are not matched...
unit_table.loc[not_matched, 'tuning_type']= "not_matched"
#I will change the label to make it coincide with our previous version of the code... TODO maybe updating the code would be a cool idea
unit_table.loc[unit_table['tuning_type']=='non_selective', 'tuning_type']= "not_selective"
#I will eliminate all the non-used columns from v3 and results of the units-funcprop merge...
unit_table.drop(columns=['pref_ori', 'gOSI', 'pref_dir', 'gDSI', 'target_id', "circvar_ori"], inplace=True)

#Set all the NaN pref oris to 0 to avoid problems
unit_table.loc[unit_table['tuning_type']=='not_matched', 'pref_ori_func'] = 0.
unit_table.loc[unit_table['pref_ori_func'].isna(), 'pref_ori_func'] = 0. 

#ow fill the pref ori in our selected column TODO do this directly
matched_ale = unit_table['tuning_type'] == 'selective'
unit_table.loc[:, 'pref_ori'] = 0. 
unit_table.loc[matched_ale, 'pref_ori'] = (unit_table.loc[matched_ale, 'pref_ori_func'] * 8 / np.pi).round()
unit_table['pref_ori'] = unit_table['pref_ori'].astype(int)
unit_table.drop(columns='pref_ori_func', inplace=True)

unit_table['cell_type'] = unit_table['classification_system'].apply(lambda x: x[:3])
unit_table.drop(columns=['id', 'classification_system', 'classification_system', 'brain_area'], inplace=True)
unit_table

#Set the naming of proofreading columns as we have in our current table
unit_table.rename(columns = {"strategy_dendrite" : "dendr_proof", "strategy_axon" : "axon_proof"}, inplace=True)

unit_table.loc[unit_table['axon_proof']=='none', 'axon_proof'] = 'non'
unit_table.loc[(unit_table['axon_proof']=='axon_interareal')|(unit_table['axon_proof']=='axon_partially_extended'), 'axon_proof'] = 'clean'
unit_table.loc[unit_table['axon_proof']=='axon_fully_extended', 'axon_proof'] = 'extended'

unit_table.loc[unit_table['dendr_proof']=='none', 'dendr_proof'] = 'non'
unit_table.loc[(unit_table['dendr_proof']=='dendrite_clean'), 'dendr_proof'] = 'clean'
unit_table.loc[(unit_table['dendr_proof']=='dendrite_extended'), 'dendr_proof'] = 'extended'

#Change name of several columns and finally select the ones we need for our unit table
unit_table.rename(columns={'pt_position_x':'pial_dist_x', 'pt_position_y':'pial_dist_y', 'pt_position_z':'pial_dist_z'}, inplace=True)
unit_table = unit_table[['pt_root_id', 'cell_type', 'tuning_type', 'layer', 'axon_proof', 'dendr_proof', 'pref_ori', 'pial_dist_x', 'pial_dist_y', 'pial_dist_z']]

#Save the result
unit_table.to_csv("data/preprocessed/unit_table_v1300.csv")

# ------------ Download synapse data ----------------

if args['download_synapses']:
    preunits = unit_table.loc[(unit_table['axon_proof'] != 'non') & (unit_table['layer'].isin(['L23', 'L4'])), 'pt_root_id'].values
    postunits = unit_table.loc[unit_table['layer']=='L23', 'pt_root_id'].values

    cleaner.download_synapse_data(presynaptic_set=preunits, postsynaptic_set=postunits, drop_synapses_duplicates=True)

# ------------ Format synapse table ----------------

cleaner.merge_synapses(syn_table_name="connections_table_v1300")
synapses = pd.read_csv('data/1300/raw/connections_table_v1300.csv')
synapses.drop(columns=["Unnamed: 0"], inplace=True)
synapses.rename(columns={'size':'syn_volume'}, inplace=True)
synapses.to_csv("data/preprocessed/connections_table_v1300.csv")

# ----------- Format activity table ---------------------

activity = pd.read_csv("data/in_processing/activity_table.csv")
activity['target_id'] = activity['target_id'].astype(int)

#Merge and get only the areas we are interested in
activity_merged = units.merge(activity, left_on='id', right_on='target_id', how='inner')
activity_merged = activity_merged[['pt_root_id', 'brain_area', 'layer', 'angle_shown', 'resp', 'sem']]

#Filter for V1 L23 and L4 TODO use the filters
activity_merged = activity_merged[(activity_merged['brain_area']=='V1')&(activity_merged['layer'].isin(['L2/3', 'L4']))]

#Then just get the minimal columns needed
activity_merged = activity_merged[['pt_root_id', 'angle_shown', 'resp', 'sem']]

#Rename to match columns to our codebase and save
activity_merged.rename(columns={'pt_root_id':'neuron_id', 'resp':'rate'}, inplace=True)

activity_merged.to_csv("data/preprocessed/activity_table_v1300.csv")