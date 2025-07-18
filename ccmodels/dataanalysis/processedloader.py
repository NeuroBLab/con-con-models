import pandas as pd
import numpy as np

import ccmodels.dataanalysis.filters as fl
import ccmodels.utils.angleutils as au
import ccmodels.dataanalysis.utils as utl

#============================================================
# ---------------------- LOAD UTILITIES ---------------------
#
# Functions here help to load the preprocessed data 
#============================================================

def load_data(orientation_only=True, nangles=16, prepath="../con-con-models/data/", suffix="", version="661"):
    """
    Load the neurons and the connections. If activity is true, also returns the activity as a Nx16 array.
    All returned values are inside a 3-element list.

    Parameters
    activity: bool 
        If True (default), returns also a Nxnangles matrix containing the rates of the neurons
    half_angle : bool 
        If True (default) uses angles only from 0 to pi, leaving apart the directions. 
    nangles: int
        Gives the number of angles for the full case (default 16)
    path : string.
        Folder where the ccmodels package is located 
    version : string
        Which version of the dataset to use.
    """

    v1_neurons = pd.read_csv(f'{prepath}/preprocessed/unit_table_v1300{suffix}.csv')
    v1_connections = pd.read_csv(f'{prepath}/preprocessed/connections_table_v1300{suffix}.csv')
    rates_table = pd.read_csv(f'{prepath}/preprocessed/activity_table_v1300{suffix}.csv')


    #Sort with the selective ones first in order to match the ids in activity table
    v1_neurons = v1_neurons.sort_values(by='tuning_type', ascending=False).reset_index(drop=False)

    #Ensure all ids are from 0 to N-1, being N number of neurons. 
    #Rename id names.
    remap_all_tables(v1_neurons, v1_connections, rates_table)



    #Get the matrix only for functionally matched neurons
    func_matched_neurons = fl.filter_neurons(v1_neurons, tuning="matched")
    rates = get_rates_matrix(func_matched_neurons, rates_table)

    #v1_neurons.loc[func_matched_neurons['id'], 'pref_ori'] = np.argmax(rates, axis=1)
    #v1_neurons.loc[v1_neurons['id'].isin(func_matched_neurons['id']), 'pref_ori']= np.argmax(rates, axis=1)

    #Angles are integers to avoid any roundoff error
    v1_neurons.loc[:, "pref_ori"] = v1_neurons["pref_ori"].astype("Int64")

    #Once angles have been constrained, construct the delta ori values 
    v1_connections["delta_ori"] = au.construct_delta_ori(v1_neurons, v1_connections, half=orientation_only)
    v1_connections["delta_ori"] = v1_connections["delta_ori"].astype("Int64")

    return v1_neurons, v1_connections, rates


#============================================================
# ---------------------- TABLE REMAPPING---------------------
#
# To get rid of the pt_root_ids and order all our dataset from
# 0 to N-1, changing the indices on all tables 
#============================================================

def get_id_map(v1_neurons):
    """
    Returns a dictionary that allows one to change the system's ids to
    to integer numbers starting at 0
    """

    #ids_original = v1_neurons["id"]
    #ids_swap     = pd.Series(ids_original.index.values, index=ids_original)  
    #return ids_swap.to_dict()

    #"""
    N = len(v1_neurons)

    #Declare new indices, and get the current ones
    idx_reset = {} 
    ids_original = v1_neurons["id"].values

    #Dictionary mapping the current indices to new ones
    for i in range(N):
        idx_reset[ids_original[i]] = i

    return idx_reset
    #"""

def remap_table(idx_remap, table, columns):
    """
    Remap the table to have all IDs using unique integer numbers from
    0 to the number of neurons minus one. [0, N-1]. 

    Parameters:
    ==========
    idx_remap : dict 
        a dictionary in the format idx_remap[pt_root_id] = new_id. Can be obtained
        from get_id_map.
    table : DataFrame T
        The table to be remapped  
    columns : list 
        A list with the names of the columns that need to be remaped
    """

    #Perform the remapping by mappling the dictionary to the 
    #corresponding columns
    table.loc[:, columns] = table[columns].map(idx_remap.get)

def remap_all_tables(v1_neurons, v1_connections, v1_activity):
    """
    Perform the remap of all the three tables: neurons, connections and activity.
    This is needed for further processing. 

    Parameters
    ==========

    The tables to be remapped.
    """

    v1_neurons.rename(columns={"pt_root_id":"id"}, inplace=True)
    v1_activity.rename(columns={"neuron_id":"id"}, inplace=True)
    v1_connections.rename(columns={"pre_pt_root_id":"pre_id", "post_pt_root_id":"post_id"}, inplace=True)

    #Get a dictionary matchking the new ids with the pt_root ones 
    idx_remap = get_id_map(v1_neurons)

    #Remap the tables
    remap_table(idx_remap, v1_connections, ["pre_id", "post_id"])
    remap_table(idx_remap, v1_activity, ["id"])
    remap_table(idx_remap, v1_neurons, ["id"])


#============================================================
# ------------------- TABLE REMAPPING -----------------------
#
# Construct some useful matrices that will be used in the data
# analysis. 
#============================================================

#TODO adjacency matrix should not be used at this stage, so this function
#might just move to the model or not appear...
def get_adjacency_matrix(v1_neurons, v1_connections):
    """
    Returns the weightedconnectivity matrix from neurons and connections.
    """

    #Number of neurons
    N = len(v1_neurons)

    #Fill the weighted matrix
    vij = np.zeros((N,N))
    for i,j,v in v1_connections[["post_id", "pre_id", "syn_volume"]].values:
        i,j = int(i), int(j)
        vij[i,j] = v
    
    return vij

def get_adjacency_matrix2(v1_neurons, v1_connections):
    """
    Returns the weightedconnectivity matrix from neurons and connections.
    """

    #Number of neurons
    N = len(v1_neurons)

    #Fill the weighted matrix
    vij = np.zeros((N,N))
    for i,j,v in v1_connections[["post_id", "pre_id", "syn_volume"]].values:
        vij[i,j] += v
    
    return vij

def get_rates_matrix(v1_neurons, v1_activity, nangles=8):
    """
    Get a matrix in which the i-th row contains the i-th rate as a function of the angle, i.e., r(theta).
    The matrix is then N*nangles.  
    """

    #Number of neurons
    N = len(v1_neurons)

    #Fill the rates
    rates = np.empty((N, nangles))
    for i in range(N):
        rates[i,:] = v1_activity.loc[v1_activity["id"] == i, "rate"].values 

    return rates
