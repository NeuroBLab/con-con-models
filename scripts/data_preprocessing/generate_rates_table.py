import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import sys
sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
import ccmodels.preprocessing.rawloader as rawload
import ccmodels.utils.angleutils as au

#Read in data with selectivity infromation
#l234_orifits = pd.read_pickle('../../data/in_processing/orientation_fits.pkl')
l234_orifits = rawload.read_table("orientation_fits") 
units = pd.read_csv("data/preprocessed/unit_table.csv") 

l234_clean = l234_orifits[['root_id', 'activity', 'orientations' ]].drop_duplicates('root_id')
l234_clean = l234_clean.sort_values("root_id")

#Extract the root_id
#ids = np.array(l234_clean['root_id'])
#ids_reps = np.repeat(ids, 16)
#TODO filtering for now because we are not matching all neurons in l234 orifits
#but probably that's wrong. So let's check how it goes.
ids = units.loc[units["tuning_type"] != "not_matched",  "pt_root_id"]
ids_reps = np.repeat(ids, 16)
l234_clean = l234_clean[l234_clean["root_id"].isin(ids)]

#Extract orientations shown in the visual stimuli
oris = l234_clean['orientations'].values
oris_ravelled = np.array(list(oris)).ravel()
angles_indexes = np.array(list(map(au.angle_indexer, oris_ravelled)))

#Extract corresponding activity rate
acts = l234_clean['activity'].values
acts_ravelled = np.array(list(acts)).ravel()

activity_table = pd.DataFrame({'neuron_id': ids_reps,
                       'angle_shown':angles_indexes,
                       'rate': acts_ravelled })

activity_table.to_csv('data/preprocessed/activity_table.csv', index=False)