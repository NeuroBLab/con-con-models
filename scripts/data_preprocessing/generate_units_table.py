''''This script is used to assign the selectivity type, layer, OSI, proofreading status, position, cell type 
to functionally matched L2/3/4 V1 neurons and inhibitory L2/3 V1 neurons.'''

import sys
sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
import numpy as np
import pandas as pd
from standard_transform import minnie_transform_vx
import ccmodels.preprocessing.utils as utl
import ccmodels.preprocessing.connectomics as conn
import ccmodels.preprocessing.region_classifier as rcl
import ccmodels.preprocessing.rawloader as loader
import ccmodels.utils.angleutils as au

# --- Get inhibitory/excitatory neurons for the whole list of neurons we have 

print("Find region for each neuron using a forest classifier...")

#First of all, generate a table tagging exc/inh neurons.
#The table contains the ids of all objects which are neurons and what is their type.
ei_info = conn.obtain_ei_table()

#Train a classifier to see in which brain area a certain object is.
#The classifier is based on functionally matched neurons
region_classifier, region_encoder = rcl.train_classifier()

#Use the classifier to get the brain region of all of our neurons
ei_info["brain_region"] = rcl.predict(ei_info, region_classifier, region_encoder)

#Save the result
ei_info.to_csv("data/in_processing/ei_table.csv", index=False)



# --- Get functional properties in L2/3 and L4 neurons in V1

print("Get functional properties, such as OSI...")

#Read in data with selectivity information 
#l234_orifits = pd.read_pickle('../../data/in_processing/orientation_fits.pkl')
l234_orifits = loader.read_table("orientation_fits") 

# Identify the tuning types of neurons based on pvalue and r squared thresholds
l234_orifits = utl.tuning_labeler(l234_orifits, model_col = 'tuning_type')

# Subset only the neurons in L2/3 and 4 of V1 to get layer, area and position labels
func_neurons = conn.get_func_match_subset_v1l234()

#split column with position values in to three separate columns
#func_neurons[['x_pos', 'y_pos', 'z_pos']] = func_neurons['pial_distances'].apply(lambda x: pd.Series(x))


#Fit the orientation fits (which are in L234) to the neurons in L234
l234_orifits = l234_orifits.merge(func_neurons, left_on=['root_id', 'session', 'scan_idx', 'cell_id'], right_on = ['pt_root_id', 'session', 'scan_idx', 'unit_id'], how = 'inner')

# Calculate the OSI of neurons
arr_dirs = np.arange(16) * np.pi / 8.  

osis = []
#for pref_ori, tunig_type, responses in zip(l234_orifits['phi'].values, l234_orifits['tuning_type'].values, l234_orifits['activity']):
for id, pref_ori, tuning_type, responses in l234_orifits[['root_id', 'phi', 'tuning_type', 'activity']].values:

       #Find the least preferred orientation depending on the model type
       #least_pref_ori_old = utl.min_act_old(pref_ori, tuning_type, arr_dirs)


       #Calculate the OSI
       #osi_old = utl.osi_calculator_old(least_pref_ori_old, pref_ori, responses, arr_dirs)

       #TODO need to be fixed form the table itself, this is a workaround
       pref_ori_new = int(pref_ori * 8 / np.pi)

       #Find the least preferred orientation depending on the model type
       least_pref_ori = utl.min_act(pref_ori_new, tuning_type)

       #Calculate the OSI
       osi = utl.osi_calculator(least_pref_ori, pref_ori_new, responses, arr_dirs)

       #print(tuning_type, pref_ori, pref_ori_new, least_pref_ori_old * 8 / np.pi, least_pref_ori)
       #print(osi_old, osi)

       osis.append(osi)

#Assign new osi column
l234_orifits['osi'] = osis

#Add indexed version of the preferred angle
l234_orifits['pref_ori'] = au.angle_indexer(l234_orifits["phi"]) 
l234_orifits['cell_type'] = 'exc'



# --- Get not-functionally matched neurons in L2/3 and L4 in V1, and merge with the functional ones

print("Mask data for V1, L23/4 and merge tables...")

#First, from all the table, just get L2/3 and L4 in V1
mask_region = ei_info["brain_region"] == "V1"
mask_layer = (ei_info["layer"] == "L23") | (ei_info["layer"] == "L4") 
v1l234_neurons = ei_info[mask_region & mask_layer]

#Drop any neurons that are inhibitory and in L4
#inh_in_L4 = (v1l234_neurons["layer"]=="L4") & (v1l234_neurons["cell_type"] == "inh")
#v1l234_neurons = v1l234_neurons[~inh_in_L4]

#Outer mode: if a neuron is on v1 and not in orifits, just fill the orifits with nans
#In contrary case, fill v1 columns with nans, except for cell_type and layer (which will coincide)
full_table = v1l234_neurons.merge(l234_orifits, on=["pt_root_id", "cell_type", "layer"], how="outer") 

#For the neurons that weren't in v1l234 but in orifits, just fill the position manually, because 
#being a list, it cannot be put on the "on" field on the previous merge
only_in_orifits = full_table["pial_distances_x"].isna()
full_table.loc[only_in_orifits, "pial_distances_x"] = full_table.loc[only_in_orifits, "pial_distances_y"] 
del only_in_orifits #free memory

#Tag all the values that are not functionally matched 
full_table.loc[full_table["tuning_type"].isna(), "tuning_type"] = "not_matched"


# --- The inhibitory classification sometimes fails. A blatant case is when a functionally matched
#     neuron is classified as inhibitory. We drop these cases.
#     This is done by putting the exc before the inh and keeping only the first. 

full_table = full_table.sort_values(by="cell_type").drop_duplicates(subset="pt_root_id", keep="first") 




# --- Add proofreading information to the table

print("Add information on neuron proofreading")

proofread = loader.read_table('proofreading')
full_table = conn.identify_proofreading_status(full_table, proofread)


# --- Finally, the dataframe is saved!!

print("Final touches and write to disk.")

#Select some columns and rename the not-so-clear ones, coming from merges and so on
full_table = full_table[["pt_root_id", "cell_type", "tuning_type", "layer", "pial_distances_x", "status_axon", "status_dendrite", "pref_ori", "osi"]]
full_table = full_table.rename(columns={"pt_rood_id":"root_id", "pial_distances_x":"pial_distances",
                                        "status_axon":"axon_proof", "status_dendrite":"dendr_proof"})

#Set the adequate type for some of the table columns
#Take into account that this categories is affected by the order (and its important just below)
full_table["tuning_type"] = pd.Categorical(full_table["tuning_type"], categories=["direction", "orientation", "not_selective", "not_matched"])

#Sort so the functionally matched appear first. This makes easier to construct the rate table. 
#This works because the values are not sorted alphabetically but in the way done by the categories
full_table = full_table.sort_values(by="tuning_type")


full_table.to_csv('data/preprocessed/unit_table.csv', index = False)

print("Finished.")
