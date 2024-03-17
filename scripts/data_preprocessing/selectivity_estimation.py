''''This script extracts the oreintation and direction tuning properties of neurons in Layer 2/3 and 4 or V1
    NOTE: run this within the scripts folder'''

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

############# Select all cells that are NOT selective ###############

#select all those withp value larger than 0.05
not_sel = l234_orifits[l234_orifits['pvalue']>0.05]
#group by root id and select ids of only those cells that are not significant for both orientation and direction
not_sel_grouped = not_sel.groupby('root_id').count().reset_index()
not_sel_id = not_sel_grouped[not_sel_grouped['pvalue']>1]['root_id']

#Sleect only those cells that are not significant to both orientation and direction
not_sel = not_sel[not_sel['root_id'].isin(not_sel_id)]

#Drop duplicates, so the fact that there are two entries for each cell
not_sel = not_sel.drop_duplicates(subset='root_id')
not_sel['model_type'] = not_sel['model_type'].replace('single', 'not_selective')


############# Select all cells that ARE selective ###############

good = l234_orifits[l234_orifits['pvalue']<0.05]

#Select all cells that are significant according to both models
grouped_res = good.groupby(['root_id']).count().reset_index()
double_sig = grouped_res[grouped_res['pvalue']>1]['root_id'].values

#Select only the double model for those with 'fringe case 1'
double_fringe = good[(good['root_id'].isin(double_sig)) & (good['r_squared_diff']>0.8) & (good['model_type'] == 'double')]

#Select only single model in neurons where both models significant and not in 'fringe case 1'
single_good = good[(good['root_id'].isin(double_sig)) & (good['r_squared_diff']<0.8) & (good['model_type'] == 'single')]

#Select all remaining neurons with only one significant model
remaining_good = good[~good['root_id'].isin(double_sig)]

almost_good = pd.concat([double_fringe,single_good,remaining_good])

#select all neurons from 'fringe case 2'
fringe2 = almost_good[(almost_good['r_squared_diff']<-0.5)&(almost_good['model_type'] == 'double') & (almost_good['pvalue'] <0.05)]['root_id']

all_good = almost_good[~almost_good['root_id'].isin(fringe2)]

all_good['model_type'] = all_good['model_type'].replace('single', 'direction')
all_good['model_type'] = all_good['model_type'].replace('double', 'orientation')

#Concatenate selective and non selective cells
neur_seltype = pd.concat([all_good, not_sel])
neur_seltype = neur_seltype.iloc[:, [0,1,2,3,5,9,11,12]]

#Merge to get information on layer
neur_seltype = neur_seltype.merge(func_neurons, left_on=['root_id', 'session', 'scan_idx', 'unit_id'], right_on = ['pt_root_id', 'session', 'scan_idx', 'unit_id'], how = 'left')
neur_seltype = neur_seltype.iloc[:, [0,1,2,3,4,5,6,7,9]]

########################################## OSI ##########################################
#Directions
arr_dirs = np.array(sorted(list(set(neur_seltype['orientations'].values[0]))))

#Find the least preferred orientation depending on the model type
least = []
for mt, pref in zip(neur_seltype['model_type'].values, neur_seltype['phi'].values):
    least.append(min_act(pref, mt, arr_dirs))

neur_seltype['least_po'] = least

# Extract the activity at the preferred orientation and at the least preferred one
maxact = []
minact = []
for ma, mi, act in zip(neur_seltype['phi'], neur_seltype['least_po'],neur_seltype['activity']):
    ind_max = np.where(arr_dirs==ma)[0]
    ind_min = np.where(arr_dirs==mi)[0]
    maxact.append(act[ind_max[0]])
    minact.append(act[ind_min[0]])

#Assigne the new columns and calculate the osi
neur_seltype['act_po'] = maxact
neur_seltype['act_lpo'] = minact
neur_seltype['osi'] = (neur_seltype['act_po']-neur_seltype['act_lpo']).values/(neur_seltype['act_po']+neur_seltype['act_lpo']).values