import pandas as pd
import numpy as np

import os
import sys
import argparse

from scipy.stats import false_discovery_control

sys.path.append(os.getcwd())
import ccmodels.utils.angleutils as au

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

#Read the result of the functional fits and chekc the coregistrated neurons
#funcprops = pd.read_csv("data/in_processing/functional_fits_validonly.csv")
funcprops = pd.read_csv("data/in_processing/functional_fits_nobound.csv")
coreg = pd.read_csv("data/1300/raw/coregistration_manual_v4.csv") 

dt = pd.read_csv("data/1300/raw/functional_properties_v3_bcm.csv")

#"""
#Inner merge to make sure we are selecting only the coregistrated neurons. Consider only the properties of the funcprops table
funcp_coreg = coreg.merge(funcprops, on=['session', 'scan_idx', 'unit_id'], how='inner')[['target_id'] + list(funcprops.columns)] 
#funcp_coreg = funcp_coreg[~((funcp_coreg['session'] == 7)&(funcp_coreg['scan_idx'] == 4))]
#funcp_coreg = funcp_coreg[funcp_coreg['scan_idx'] != 4]

#funcp_coreg = funcp_coreg[(funcp_coreg['session'] == 9)]

#This table still can have several scans/sessions referring to the same units. We need disambiguate those. 
#To do that, we will take only those neurons which have consistent results across multiple scans. 
#Just compute the spread across ocurrences of the pref ori. If the neuron appears only once, spread is 0.
grouped = funcp_coreg.groupby('target_id')['pref_ori'].agg(lambda x: au.angle_dist(x.max(), x.min()))

#Select neurons with low spread (high consistency across scans or there was a single scan)
idx_selected = grouped[grouped.abs() <= 1].index.values
mask_selected = funcp_coreg['target_id'].isin(idx_selected)

#Sort these values by decreasing r2. Then drop duplications on target id, keeping only the first element: the one with highest r2
filtered_func = funcp_coreg[mask_selected].sort_values(by=['target_id', 'r2_ori'], ascending=False)
filtered_func = filtered_func.drop_duplicates(subset='target_id', keep='first')

#Use the table to set up a criteria for selectivity. 
#Since we used indepedent multiple comparisons, we can employ the Benjamini-Hochberg method (from scipy) to set FDR to 0.05
filtered_func['tuning_type'] = 'not_selective'
pvals_rescaled = false_discovery_control(filtered_func['pvals_ori'], method='bh')
filtered_func.loc[(filtered_func['r2_ori'] > 0.5) & (pvals_rescaled < 0.01), 'tuning_type'] = 'selective'
print(f'selective {len(filtered_func[filtered_func["tuning_type"]=="selective"])}')
#filtered_func['tuning_type'] = 'selective'

#Filter the columns we will actually use 
filtered_func['target_id'] = filtered_func['target_id'].astype(int)
#"""

"""
funcp_coreg = dt.merge(funcprops, on=['session', 'scan_idx', 'unit_id'], how='inner') 
funcp_coreg = funcp_coreg[['target_id', 'session', 'scan_idx', 'unit_id', 'pref_ori_x', 'pref_ori_y', 'rate_ori', 'r2_ori', 'cc_abs']]
funcp_coreg['pref_ori'] = np.round(funcp_coreg['pref_ori_x'] * 8 / np.pi).astype(int)
funcp_coreg.loc[:, 'pref_ori'] = funcp_coreg.loc[:, 'pref_ori'] % 8 

funcp_coreg = funcp_coreg.sort_values(by='cc_abs', ascending=False)
filtered_func = funcp_coreg.drop_duplicates(subset='target_id', keep='first')

#grouped = funcp_coreg.groupby('target_id')['pref_ori'].agg(lambda x: au.angle_dist(x.max(), x.min()))
#idx_selected = grouped[grouped.abs() <= 1].index.values
#mask_selected = funcp_coreg['target_id'].isin(idx_selected)

#filtered_func = funcp_coreg[mask_selected].sort_values(by=['target_id', 'r2_ori'], ascending=False)
#filtered_func = filtered_func.drop_duplicates(subset='target_id', keep='first')

filtered_func['tuning_type'] = 'selective'
print(len(filtered_func))
"""
func_4_units = filtered_func[['target_id', 'pref_ori', 'tuning_type']]

# ------------ Format nucleus table with functional stuff ----------------

print("Generate unit table...")

#Process the data and obtain units and segment tables
units, segments = cleaner.process_nucleus_data(with_functional=False)
units.loc[units['layer']=='L2/3', 'layer'] = 'L23'

#These 3 neurons are found in the coregistration table so they should be excitatory, but the AIBS classification say they are nonneuronal.
#Since the coregistration is manual, we favor that one and set the neurons manually to be excitatory so when we filter they do not disappear
units.loc[(units['pt_root_id'].isin([864691135726289983,864691135569616364,864691136085014636])), 'classification_system'] = 'excitatory_neuron'

print(len(units['pt_root_id']), len(units['pt_root_id'].unique()))

#Merge the unit table with our matched functional properties
units_with_func = units.merge(func_4_units, left_on='id', right_on='target_id', how='left')
units_with_func = units_with_func.drop(columns='target_id') 

print(len(units_with_func['pt_root_id']), len(units_with_func['pt_root_id'].unique()))

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
activity = filtered_func[['target_id', 'rate_ori']]
activity.loc[:, 'rate_ori'] = activity['rate_ori'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')) 
activity = activity.explode('rate_ori')
#Write the angle that corresponds to each one 
activity['angle_shown'] = np.tile(np.arange(8),  len(activity)//8)

#Merge and get only the areas we are interested in
activity_merged = units.merge(activity, left_on='id', right_on='target_id', how='inner')
activity_merged = activity_merged[['pt_root_id', 'brain_area', 'layer', 'angle_shown', 'rate_ori']]

#Filter for V1 L23 and L4 TODO use the filters
activity_merged = activity_merged[(activity_merged['brain_area']=='V1')&(activity_merged['layer'].isin(['L23', 'L4']))]

#Then just get the minimal columns needed
activity_merged = activity_merged[['pt_root_id', 'angle_shown', 'rate_ori']]

#Rename to match columns to our codebase and save
activity_merged.rename(columns={'pt_root_id':'neuron_id', 'rate_ori':'rate'}, inplace=True)

activity_merged.to_csv(f"data/preprocessed/activity_table_v1300{table_suffix}.csv", index=False)