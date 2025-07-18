import pandas as pd
import numpy as np

import os
import sys
import argparse

from scipy.stats import false_discovery_control

sys.path.append(os.getcwd())
#import ccmodels.utils.angleutils as au

import microns_datacleaner as mic

# ----------------------- User input ----------------------------------

parser = argparse.ArgumentParser(description='''Data download and processing''')

# Adding and parsing arguments
parser.add_argument('--download_nucleus',  action='store_true', help='Boolean. If true, downloads all nucleus data')
parser.add_argument('--download_synapses', action='store_true', help='Boolean. If true, downloads all the synapse data as well')
parser.add_argument('--table_suffix', default="") 
args = parser.parse_args()

table_suffix = str(args.table_suffix)

#Add a leading lower dash if used
if len(table_suffix) > 0:
    table_suffix = "_" + table_suffix

# -------------------- Nucleus download --------------------------

cleaner = mic.MicronsDataCleaner(datadir="data", version=1300, download_policy='minimum')

#Download all data related to nucleus: reference table, various classifications, functional matchs
if args.download_nucleus:
    print("Download nucleus data")
    cleaner.download_nucleus_data()

# ----------- Format functional data ------------------------ 

print("Format functional data...")

"""
#DIGITAL TWIN - NO ACTIVITY
funcprops = pd.read_csv("data/in_processing/functional_fits_nobound.csv")
dt = pd.read_csv("data/1300/raw/functional_properties_v3_bcm.csv")
dt['pref_ori'] = np.round(dt['pref_ori'] * 8 / np.pi).astype(int)
dt.loc[:, 'pref_ori'] = dt.loc[:, 'pref_ori'] % 8 


dt = dt.sort_values(by='cc_abs', ascending=False)
dt = dt.drop_duplicates(subset='target_id', keep='first')

dt = dt[dt['pt_root_id'] != 0]

dt['tuning_type'] = 'selective'

func_4_units    = dt[['target_id', 'pref_ori', 'tuning_type']]
"""

#Read the result of the functional fits and chekc the coregistrated neurons
coreg = pd.read_csv("data/1300/raw/coregistration_manual_v4.csv") 
funcprops = pd.read_csv("data/in_processing/functional_fits_nobound.csv")

coreg = coreg[['session', 'scan_idx', 'unit_id', 'target_id']]

#funcprops = coreg.merge(funcprops, on=['session', 'scan_idx', 'unit_id'], how='left')
#funcprops = funcprops[funcprops['pref_ori'].notna()].reset_index(drop=True)
funcprops = coreg.merge(funcprops, on=['session', 'scan_idx', 'unit_id'], how='inner')

funcprops = funcprops.sort_values(by='r2_ori', ascending=False)
funcprops = funcprops.drop_duplicates(subset='target_id', keep='first')

funcprops['tuning_type'] = 'not_selective'
pvals_rescaled = false_discovery_control(funcprops['pvals_ori'], method='bh')
funcprops.loc[(funcprops['r2_ori'] > 0.5) & (pvals_rescaled < 0.01), 'tuning_type'] = 'selective'
funcprops['target_id'] = funcprops['target_id'].astype(int)

func_4_units = funcprops[['target_id', 'pref_ori', 'tuning_type']]
activity     = funcprops[['target_id', 'rate_ori']]


# ------------ Format nucleus table with functional stuff ----------------

print("Generate unit table...")

#Process the data and obtain units and segment tables
units, segments = cleaner.process_nucleus_data(with_functional='no')
units.loc[units['layer']=='L2/3', 'layer'] = 'L23'

#These 3 neurons are found in the coregistration table so they should be excitatory, but the AIBS classification say they are nonneuronal.
#Since the coregistration is manual, we favor that one and set the neurons manually to be excitatory so when we filter they do not disappear
units.loc[(units['pt_root_id'].isin([864691135726289983,864691135569616364,864691136085014636])), 'classification_system'] = 'excitatory_neuron'


#Merge the unit table with our matched functional properties
units_with_func = units.merge(func_4_units, left_on='nucleus_id', right_on='target_id', how='left')
units_with_func = units_with_func.drop(columns='target_id') 


#Filter for our brain area and layer of interest
#TODO use the filters!
unit_table = units_with_func.loc[(units_with_func['brain_area']=='V1')&(units_with_func['layer'].isin(['L23', 'L4']))&(units_with_func['classification_system']!='nonneuron'), :]

print(len(unit_table['pt_root_id']), len(unit_table['pt_root_id'].unique()))

#Get the neurons not matched in coreg v4:
not_matched = unit_table['tuning_type'].isna()

#Label them as not matched and fix its pref ori to a (meaningless) value so there is no more nans
unit_table.loc[not_matched, 'tuning_type']= "not_matched"
unit_table.loc[not_matched, 'pref_ori']= 0.
#Once there is no more nans, we can put them as integers
unit_table['pref_ori'] = unit_table['pref_ori'].astype(int)

#Use 'exc' or 'inh' as the cell type, using the first three characters from the classification system. 
#Then drop all the columns that we will not need
unit_table['cell_type'] = unit_table['classification_system'].apply(lambda x: x[:3])
unit_table.drop(columns=['classification_system', 'classification_system', 'brain_area'], inplace=True)

#Set the naming of proofreading columns as we have in our current table
unit_table.rename(columns = {"id":"nucleus_id", "strategy_dendrite" : "dendr_proof", "strategy_axon" : "axon_proof"}, inplace=True)

unit_table.loc[unit_table['axon_proof']=='none', 'axon_proof'] = 'non'
unit_table.loc[(unit_table['axon_proof']=='axon_interareal')|(unit_table['axon_proof']=='axon_partially_extended'), 'axon_proof'] = 'clean'
unit_table.loc[unit_table['axon_proof']=='axon_fully_extended', 'axon_proof'] = 'extended'

unit_table.loc[unit_table['dendr_proof']=='none', 'dendr_proof'] = 'non'
unit_table.loc[(unit_table['dendr_proof']=='dendrite_clean'), 'dendr_proof'] = 'clean'
unit_table.loc[(unit_table['dendr_proof']=='dendrite_extended'), 'dendr_proof'] = 'extended'

#Change name of several columns and finally select the ones we need for our unit table
unit_table.rename(columns={'pt_position_x':'pial_dist_x', 'pt_position_y':'pial_dist_y', 'pt_position_z':'pial_dist_z'}, inplace=True)
unit_table = unit_table[['pt_root_id', 'nucleus_id', 'cell_type', 'tuning_type', 'layer', 'axon_proof', 'dendr_proof', 'pref_ori', 'pial_dist_x', 'pial_dist_y', 'pial_dist_z']]

print(len(unit_table), unit_table['nucleus_id'].value_counts().sum(), unit_table['pt_root_id'].value_counts().sum())
print(len(unit_table['pt_root_id']), len(unit_table['pt_root_id'].unique()))

#Save the result
unit_table.to_csv(f"data/preprocessed/unit_table_v1300{table_suffix}.csv", index=False)

# ------------ Download synapse data ----------------

if args.download_synapses:
    print("Download synapses...")
    preunits  = unit_table.loc[(unit_table['axon_proof'] != 'non') & (unit_table['layer'].isin(['L23', 'L4'])), 'pt_root_id'].values
    postunits = unit_table.loc[unit_table['layer']=='L23', 'pt_root_id'].values

    cleaner.download_synapse_data(presynaptic_set=preunits, postsynaptic_set=postunits, drop_synapses_duplicates=True)

# ------------ Format synapse table ----------------

print("Merge synapses...")
cleaner.merge_synapses(syn_table_name="connections_table_v1300")
synapses = pd.read_csv('data/1300/raw/connections_table_v1300.csv')
#synapses.drop(columns=["Unnamed: 0"], inplace=True)
synapses.rename(columns={'size':'syn_volume'}, inplace=True)
synapses.to_csv(f"data/preprocessed/connections_table_v1300{table_suffix}.csv", index=False)

# ----------- Format activity table ---------------------

print("Generate activity table...")
#Get the activity table by expanding (exploding) the rates of the selected units
#The rate_ori column is str so we transform it to arrays first 
activity = funcprops[['target_id', 'rate_ori']]
activity.loc[:, 'rate_ori'] = activity['rate_ori'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')) 
activity = activity.explode('rate_ori')
#Write the angle that corresponds to each one 
activity['angle_shown'] = np.tile(np.arange(8),  len(activity)//8)

#Merge and get only the areas we are interested in
activity_merged = units.merge(activity, left_on='nucleus_id', right_on='target_id', how='inner')
activity_merged = activity_merged[['pt_root_id', 'brain_area', 'layer', 'angle_shown', 'rate_ori']]

#Filter for V1 L23 and L4 TODO use the filters
activity_merged = activity_merged[(activity_merged['brain_area']=='V1')&(activity_merged['layer'].isin(['L23', 'L4']))]

#Then just get the minimal columns needed
activity_merged = activity_merged[['pt_root_id', 'angle_shown', 'rate_ori']]

#Rename to match columns to our codebase and save
activity_merged.rename(columns={'pt_root_id':'neuron_id', 'rate_ori':'rate'}, inplace=True)

activity_merged.to_csv(f"data/preprocessed/activity_table_v1300{table_suffix}.csv", index=False)