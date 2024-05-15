import numpy as np
import math

import functions as fun

import random
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, ks_2samp

def statsextract(prepath='data'):
    # load files
    path_to_folder=f"{prepath}/preprocessed/"
    activity_table = pd.read_csv(path_to_folder+"activity_table.csv")
    connections_table = pd.read_csv(path_to_folder+"connections_table.csv")
    unit_table = pd.read_csv(path_to_folder+"unit_table.csv")

    # put pref orientation of not-selective neurons to nan
    mask_not_selective=(unit_table['tuning_type']=='not_selective')
    unit_table.loc[mask_not_selective, 'pref_ori'] = np.nan
    # focus on orientation, take preferred orientation mod pi
    unit_table.loc[:,'pref_ori']=np.mod(unit_table.loc[:,'pref_ori'],8)
    mask_selective=(unit_table['tuning_type']=='orientation')|(unit_table['tuning_type']=='direction')
    unit_table.loc[mask_selective, 'tuning_type'] ='selective'




    connections_table=connections_table[connections_table['pre_pt_root_id']!= connections_table['post_pt_root_id']]
    # normalize synaptic volumes with respect to average E -E connection strength

    norm_V=fun.Compute_normalization_factor_for_synaptic_volume(connections_table, unit_table)
    connections_table.loc[:,'syn_volume']=connections_table.loc[:,'syn_volume']/norm_V


    # List of all the possible labels neurons can have: areas, layer,  cell type, and tuning type
    Labels = []
    for area in ['V1']:
        for layer in ['L23', 'L4']:
            possible_cell_type = ['exc', 'inh'] if layer == 'L23' else ['exc']
            for cell_type in possible_cell_type:
                for tuning_type in ['selective','not_selective']:
                    mask_cell_type = (unit_table['cell_type'] == cell_type) & (unit_table['layer'] == layer)
        
                    label_info = {'area': area,
                                'layer': layer,
                                'cell_type': cell_type,
                                'tuning_type':tuning_type
                                }
                    # Append label information to Labels
                    Labels.append(label_info)

    #pd.to_pickle(Labels, 'data/model/aledata.pkl')
    return unit_table, connections_table, activity_table, Labels