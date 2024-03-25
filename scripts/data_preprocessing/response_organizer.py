import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ccmodels.preprocessing.utils import angle_indexer

#Read in data with selectivity infromation
l234_orifits = pd.read_pickle('../../data/in_processing/orientation_fits.pkl')

l234_clean = l234_orifits[['root_id', 'activity', 'orientations' ]].drop_duplicates('root_id')

#Extract the root_id
ids = np.array(l234_clean['root_id'])
ids_reps = np.repeat(ids, 16)

#Extract orientations shown in the visual stimuli
ors = l234_clean['orientations'].values
ors_ravelled = np.array(list(ors)).ravel()
angles_indexes = np.array(list(map(angle_indexer, ors_ravelled)))

#Extract corresponding activity rate
acts = l234_clean['activity'].values
acts_ravelled = np.array(list(acts)).ravel()

activity_table = pd.DataFrame({'neuron_id': ids_reps,
                       'angle_shown':angles_indexes,
                       'rate': acts_ravelled })

activity_table.to_csv('../../data/preprocessed/activity_table.csv', index=False)