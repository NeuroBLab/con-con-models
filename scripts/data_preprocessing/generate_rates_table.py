import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.preprocessing.rawloader as rawload
import ccmodels.utils.angleutils as au
import ccmodels.dataanalysis.filters as fl
import ccmodels.preprocessing.utils as utl

#Read in data with selectivity information and take the correct activities 
l234_orifits = rawload.read_table("orientation_fits") 
orifits = utl.tuning_labeler(l234_orifits, model_col="tuning_type")

#Get functionally matched units from the table
units = pd.read_csv("data/preprocessed/unit_table.csv") 
matched = fl.filter_neurons(units, tuning="matched")

#This merge is important to be done from orifits TO matched, not the contrary, since in this 
#way the order of the root_ids is the order of the matched neurons. The rates table shall be in 
#the same order, so the i-th functionally matched neuron of the table is also the i-th row in the matrix
merged = matched.merge(orifits, right_on=["root_id"], left_on=["pt_root_id"], how="inner")


#Repeat each id, in the order it appears, 16 times
ids_reps = np.repeat(merged["pt_root_id"].values, 16)

#Extract orientations shown in the visual stimuli
oris = merged['orientations'].values
oris_ravelled = np.array(list(oris)).ravel()
angles_indexes = np.array(list(map(au.angle_indexer, oris_ravelled)))

#Extract corresponding activity rate
acts = merged['activity'].values
acts_ravelled = np.array(list(acts)).ravel()

activity_table = pd.DataFrame({'neuron_id': ids_reps,
                       'angle_shown':angles_indexes,
                       'rate': acts_ravelled })

activity_table.to_csv('data/preprocessed/activity_table.csv', index=False)