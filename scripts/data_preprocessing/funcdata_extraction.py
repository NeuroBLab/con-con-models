import os
import gc
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from ccmodels.preprocessing.connectomics import client_version, subset_v1l234
from ccmodels.preprocessing.selectivity import orientation_extractor, von_mises, von_mises_single, is_selective, cell_area_identifiers, fpd_assignment, identify_multiscan


#Identify desired neurons, in this case v1 neurons from L234
client = client_version(661)
area_v1_neurons = cell_area_identifiers('V1')
v1l234_neur = subset_v1l234(client, table_name = 'coregistration_manual_v3', area_df = area_v1_neurons)
v1l234_neur = v1l234_neur[v1l234_neur['pt_root_id'] != 0]

#Identify neurons that have been scaned multiple vs one time only
multiscan_ids = identify_multiscan(v1l234_neur)
v1l234_singlescan = v1l234_neur[~v1l234_neur['pt_root_id'].isin(multiscan_ids)]
v1l234_multiscan = v1l234_neur[v1l234_neur['pt_root_id'].isin(multiscan_ids)]

session_scan_pairs = [(8,7), (4, 7), (5, 3), (5, 6),(5, 7),(6, 2),(6, 4),(6, 6),(6, 7),(7, 3),(7, 4),(7, 5),(8, 5), (9, 3), (9, 4), (9, 6)]

#Check if storage folders for temporary data exist, if not make them
if os.path.isdir('../data/in_processing/orientation_fits') != True:
    os.makedirs('../data/in_processing/orientation_fits')


print('Starting Extraction of Single Scan...')
for pair in tqdm(session_scan_pairs, desc = 'Session and Scan Loop'):
    #change the frame per directions according to the one used for each session and scan
    fpd = fpd_assignment(pair[0], pair[1])

    #Container with saved data
    data = []
            
    #Von Mises
    columns = ['root_id', 'session', 'scan_idx','cell_id', 'pvalue','tuning_type', 
        'r_squared_diff', 'mean_r_sqrd', 'A', 'phi', 'k', 'activity', 'orientations']

    #Subset the cells for server limitation reasons
    sub = v1l234_singlescan[(v1l234_singlescan['session'] == pair[0]) & (v1l234_singlescan['scan_idx'] == pair[1])]
        
    #loop through cells
    for i in tqdm(range(sub.shape[0]), desc = f'Extracting neurons of session: {pair[0]}, scan: {pair[1]}' ):
        unit_key = {'session':sub.iloc[i, 2], 'scan_idx':sub.iloc[i, 3], 'unit_id':sub.iloc[i, 4]}
        df = orientation_extractor(unit_key, fpd)


        #Calculating max activity for constraining von mises, for plotting data and fit
        gp = df.groupby('radians').mean().reset_index()
        max_act_rad = gp.sort_values('mean_activity', ascending = False)['radians'].values[0]
            
        #Using Von Mises
        pars_d, pcov_d = curve_fit(lambda theta, A, k: von_mises(theta, A, max_act_rad, k), df['radians'], df['mean_activity'],
                                bounds = ([-np.inf, 0.1],[np.inf, np.inf]),
                                method = 'trf')

        pars_s, pcov_s = curve_fit(lambda theta, A, k: von_mises(theta, A, max_act_rad, k), df['radians'], df['mean_activity'],
                                bounds = ([-np.inf, 0.1],[np.inf, np.inf]),
                                method = 'trf')



        # Von Mises activities
        ate_d= [von_mises(i,pars_d[0], max_act_rad,  pars_d[1]) for i in gp['radians']]
        ate_s= [von_mises_single(i,pars_s[0], max_act_rad,  pars_s[1]) for i in gp['radians']]
            
        r2d = r2_score(gp['mean_activity'], ate_d)
        r2s = r2_score(gp['mean_activity'], ate_s)

        #Calculate difference in r2 score between the two model fist to identify fringe cases
        rdiff = r2d-r2s
            
        #single
        vs, ps, max_rad = is_selective(df, max_act_rad)


        #double
        vdb, pdb, max_rad = is_selective(df, max_act_rad, single = False)
            
            
        #save data
        data.append([sub.iloc[i, 0], unit_key['session'], unit_key['scan_idx'], unit_key['unit_id'],ps,'single', rdiff, r2s, 
                        pars_d[0], max_act_rad,  pars_d[1], np.array(gp['mean_activity']), np.array(gp['radians'])])
            
        #save data
        data.append([sub.iloc[i, 0], unit_key['session'], unit_key['scan_idx'], unit_key['unit_id'],pdb,'double', rdiff,r2d, 
                        pars_s[0], max_act_rad,  pars_s[1], np.array(gp['mean_activity']), np.array(gp['radians'])])
            
    #Save the data
    data_df = pd.DataFrame(data, columns = columns)
    data_df.to_pickle(f'../data/in_processing/orientation_fits/orientations_fits_{pair[0]}_{pair[1]}.pkl')
        
    #Clean RAM
    del sub, unit_key, df, gp, pars_s, pars_d, pcov_s, pcov_d, ate_d, ate_s, r2d, r2s, rdiff, vs, ps, vdb, pdb, max_rad
    gc.collect()    

print('Extraction single scan finished')
print('Starting extraction Multiscan...')

#Container with saved data
data = []
            
#Von Mises
columns = ['root_id', 'session', 'scan_idx','cell_id', 'pvalue','tuning_type', 
        'r_squared_diff', 'mean_r_sqrd', 'A', 'phi', 'k', 'activity', 'orientations']

for mult_id in tqdm(multiscan_ids, desc = f'Extracting multiscan neurons data') :
    subset_multiscan = v1l234_multiscan[v1l234_multiscan['pt_root_id'] == mult_id]
    scan_acts = []
    
    for row in range(subset_multiscan.shape[0]):
        unit_key = {'session':subset_multiscan['session'].values[row], 'scan_idx':subset_multiscan['scan_idx'].values[row], 'unit_id':subset_multiscan['unit_id'].values[row]}
        fpd = fpd_assignment(subset_multiscan['session'].values[row], subset_multiscan['scan_idx'].values[row])
        df = orientation_extractor(unit_key, fpd)
        scan_acts.append(df)
    
    all_acts = pd.concat(scan_acts, axis = 1)

    #Calculating max activity for constraining von mises, for plotting data and fit
    gp = df.groupby('radians').mean().reset_index()
    max_act_rad = gp.sort_values('mean_activity', ascending = False)['radians'].values[0]
            
    #Using Von Mises
    pars_d, pcov_d = curve_fit(lambda theta, A, k: von_mises(theta, A, max_act_rad, k), df['radians'], df['mean_activity'],
                                bounds = ([-np.inf, 0.1],[np.inf, np.inf]),
                                method = 'trf')

    pars_s, pcov_s = curve_fit(lambda theta, A, k: von_mises(theta, A, max_act_rad, k), df['radians'], df['mean_activity'],
                                bounds = ([-np.inf, 0.1],[np.inf, np.inf]),
                                method = 'trf')



    # Von Mises activities
    ate_d= [von_mises(i,pars_d[0], max_act_rad,  pars_d[1]) for i in gp['radians']]
    ate_s= [von_mises_single(i,pars_s[0], max_act_rad,  pars_s[1]) for i in gp['radians']]
            
    r2d = r2_score(gp['mean_activity'], ate_d)
    r2s = r2_score(gp['mean_activity'], ate_s)

    #Calculate difference in r2 score between the two model fist to identify fringe cases
    rdiff = r2d-r2s
            
    #single
    vs, ps, max_rad = is_selective(df, max_act_rad)


    #double
    vdb, pdb, max_rad = is_selective(df, max_act_rad, single = False)
            
            
    #save data
    data.append([mult_id, unit_key['session'], unit_key['scan_idx'], unit_key['unit_id'],ps,'single', rdiff, r2s, 
                        pars_d[0], max_act_rad,  pars_d[1], np.array(gp['mean_activity']), np.array(gp['radians'])])
            
    #save data
    data.append([mult_id, unit_key['session'], unit_key['scan_idx'], unit_key['unit_id'],pdb,'double', rdiff,r2d, 
                        pars_s[0], max_act_rad,  pars_s[1], np.array(gp['mean_activity']), np.array(gp['radians'])])
            
#Save the data
data_df = pd.DataFrame(data, columns = columns)
#NOTE: Session, scan_idx and unit_id of these neurons refer to the values in the last row of the subset_multiscan df above
data_df.to_pickle(f'../data/in_processing/orientation_fits/multiscan_fits.pkl')
#Clean RAM
del sub, unit_key, df, gp, pars_s, pars_d, pcov_s, pcov_d, ate_d, ate_s, r2d, r2s, rdiff, vs, ps, vdb, pdb, max_rad
gc.collect()   

print('Extraction multi scan finished, saving data')

#Joining all of the DataFrames
#Loading the first file in the directory
ors_all = pd.read_pickle(f"../data/in_processing/orientation_fits/{os.listdir('../data/in_processing/orientation_fits')[0]}")

#Loading the rest iteratively and concatenating
for file in tqdm(os.listdir('../data/in_processing/orientation_fits/')[1:], desc='Aggregating session, scan files'):
    if file!='.DS_Store':
        cont_df = pd.read_pickle(f'../data/in_processing/orientation_fits/{file}')
        ors_all = pd.concat([ors_all, cont_df], axis = 0)
    else:
        continue

#Saving the data
ors_all.to_pickle('../data/in_processing/orientation_fits.pkl')

#Delete unnecessary data repeats
shutil.rmtree('../data/in_processing/orientation_fits')
