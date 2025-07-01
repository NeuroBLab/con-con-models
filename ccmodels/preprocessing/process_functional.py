import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

sys.path.append(os.getcwd())
import ccmodels.preprocessing.utils_new as ut


# ------------------------- Main function (program logic is below) ---------------------------------

#Performs a fit of the tuning curves and check if the difference between the rates at different opposed angles 
#is significant for a given session and scan. Returns a dataframe with all the information. 
def fit_tuning_curves_and_check_significance(session, scan_idx, funcdatapath="data/functional"):

    #Get the path to the main folder where this session and scan are stored
    folder_data =f"{funcdatapath}/{session}_{scan}" 

    #Load sessions's information about trials and units recorded
    trial_type  = np.load(f"{folder_data}/meta/trials/type.npy")
    valid_trial = np.load(f"{folder_data}/meta/trials/valid_trial.npy")
    unit_ids    = np.load(f"{folder_data}/meta/neurons/unit_ids.npy")

    #Get only the Monet trials
    monet_trials = np.where(trial_type == 'Monet2')[0]
    #monet_trials = np.where((trial_type == 'Monet2')&(valid_trial==True))[0]

    #Initialize our variables
    n_neurons = len(unit_ids)
    n_frames = 450 #Fixed number of frames per trial
    n_trials = len(monet_trials)
    responses = np.empty((n_neurons, n_trials, n_frames))
    directions =  np.empty((n_trials, n_frames))

    #Read the responses per frame and which frames correspond to each direction
    for k,monet_idx in enumerate(monet_trials): 
        responses[:, k, :] = np.load(f"{folder_data}/data/responses/{monet_idx}.npy")
        loaded_dirs        = np.load(f"{folder_data}/meta/trials/directions/{monet_idx}.npy")
        directions[k, :] = np.round(loaded_dirs* 16 / 360).astype(int) #Angles are integers 0-15

    #Set firing rate to 5 spk / second (Sanzeni et al 2020)
    responses *= 5. / responses.mean()

    #Initialize the average firing rates
    avgrate_dir     = np.empty((n_neurons, 16))
    avgrate_ori     = np.empty((n_neurons, 8))

    #Reshape the vectors so we have all trials one after another per each neuron
    response_stacked_trials = responses.reshape(n_neurons, n_trials*n_frames)
    dirs_stacked_trials = directions.reshape(n_trials * n_frames)
    oris_stacked_trials = dirs_stacked_trials % 8 #Define also the current orientation per frame

    #Compute average responses to direction and orientation
    for dir in range(16):
        mask_dir = dirs_stacked_trials == dir
        avgrate_dir[:, dir] = response_stacked_trials[:, mask_dir].mean(axis=1)
    for ori in range(8):
        mask_ori = oris_stacked_trials == ori
        avgrate_ori[:, ori] = response_stacked_trials[:, mask_ori].mean(axis=1)

    #Number of parameters for the fitting of von Mises function for each cases
    npars_ori = 4
    npars_dir = 5

    #Initialize observables and the result container 
    pars_ori = np.empty((n_neurons, npars_ori))
    r2_ori = np.empty(n_neurons)
    pref_ori = np.empty(n_neurons)
    pars_dir = np.empty((n_neurons, npars_dir))
    r2_dir = np.empty(n_neurons)
    pref_dir = np.empty(n_neurons)

    results = pd.DataFrame(columns=['unit_ids', 'session', 'scan_idx', 'pref_ori', 'r2_ori', 'pvals_ori', 'pref_dir', 'r2_dir', 'pvals_dir_mid', 'pvals_dir_anti']) 
    results['unit_ids'] = unit_ids
    results['session']  = session  * np.ones(n_neurons) 
    results['scan_idx'] = scan_idx * np.ones(n_neurons) 

    #fit's x-axis are the angles
    thetas_ori = np.linspace(0, np.pi, 8, endpoint=False)
    thetas_dir = np.linspace(0, 2*np.pi, 16, endpoint=False)

    #Perform all the fits for each neuron, using the average tuning curves
    for id in range(n_neurons):
        pars_ori[id, :], r2_ori[id] = ut.fit_ori(thetas_ori, avgrate_ori[id, :])
        pars_dir[id, :], r2_dir[id] = ut.fit_dir(thetas_dir, avgrate_dir[id, :])

    #Now check of the differences between preferred orientation and the mid/anti points to see if 
    #tuning is significant  
    pref_ori, pvals_ori                     = ut.test_ori(n_neurons, thetas_ori, response_stacked_trials, oris_stacked_trials, pars_ori)
    pref_dir, pvals_dir_mid, pvals_dir_anti = ut.test_dir(n_neurons, thetas_dir, response_stacked_trials, dirs_stacked_trials, pars_dir)

    #Save our results
    results['pref_ori'] = pref_ori 
    results['r2_ori'] = r2_ori
    results['pvals_ori'] = pvals_ori 

    results['pref_dir'] = pref_dir
    results['r2_dir'] = r2_dir
    results['pvals_dir_mid']  = pvals_dir_mid 
    results['pvals_dir_anti'] = pvals_dir_anti

    return results


# ------------------------- User input ---------------------------------

parser = argparse.ArgumentParser(description='''Process the functional data''')

# Adding and parsing arguments
parser.add_argument('funcdatapath', type=str, help='Path where the functional data is stored')
args = parser.parse_args()


#Create the master table
functional_table = pd.DataFrame(columns=['unit_ids', 'session', 'scan_idx', 'pref_ori', 'r2_ori', 'pvals_ori', 'pref_dir', 'r2_dir', 'pvals_dir_mid', 'pvals_dir_anti']) 

folders_to_analyse = os.listdir(args.funcdatapath) 
folders_to_analyse = ['6_6']

#Get all the pairs (session, scan) in a convenient array
session_scan_pair = []
for f in folders_to_analyse:
    session = int(f[0])
    scan    = int(f[2])
    table = fit_tuning_curves_and_check_significance(session, scan, funcdatapath=args.funcdatapath)

    functional_table = pd.concat([functional_table, table], ignore_index=True)

functional_table.to_csv("data/in_processing/functional_fits.csv")