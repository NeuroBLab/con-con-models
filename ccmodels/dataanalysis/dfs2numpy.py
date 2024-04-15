import pandas as pd
import numpy as np

def get_id_map(v1_neurons):
    """
    Returns a dictionary that allows one to change the system's ids to
    to integer numbers starting at 0
    """

    N = len(v1_neurons)

    #Declare new indices, and get the current ones
    idx_reset = {} 
    ids_original = v1_neurons["id"].values

    #Dictionary mapping the current indices to new ones
    for i in range(N):
        idx_reset[ids_original[i]] = i

    return idx_reset

def remap_table(v1_neurons, table, columns):
    """
    Remap the table to have all IDs using unique integer numbers from
    0 to the number of neurons minus one. [0, N-1]. 

    Parameters:
    ==========
    v1_neurons : DataFrame
        info about the neurons,
    table : DataFrame T
        he table to be remapped  
    columns : list 
        A list with the names of the columns that need to be remaped
    """

    #Get the indices linking neuron ids with the new ones
    idx_remap = get_id_map(v1_neurons)

    #Perform the remapping
    for c in columns:
        table[c] = table[c].map(idx_remap)

def remap_all_tables(v1_neurons, v1_connections, v1_activity):
    """
    Perform the remap of all the three tables: neurons, connections and activity.
    This is needed for further processing. 

    Parameters
    ==========

    The tables to be remapped.
    """

    v1_neurons.rename(columns={"root_id":"id"}, inplace=True)
    v1_activity.rename(columns={"neuron_id":"id"}, inplace=True)
    v1_connections.rename(columns={"pre_pt_root_id":"pre_id", "post_pt_root_id":"post_id"}, inplace=True)

    remap_table(v1_neurons, v1_connections, ["pre_id", "post_id"])
    remap_table(v1_neurons, v1_activity, ["id"])
    remap_table(v1_neurons, v1_neurons, ["id"])



def get_adjacency_matrix(v1_neurons, v1_connections):
    """
    Returns the weightedconnectivity matrix from neurons and connections.
    """

    #Number of neurons
    N = len(v1_neurons)

    #Fill the weighted matrix
    vij = np.zeros((N,N))
    for i,j,v in v1_connections[["post_id", "pre_id", "size"]].values:
        vij[i,j] = v
    
    return vij

def get_rates_matrix(v1_neurons, v1_activity, nangles=16):
    """
    Get a matrix in which the i-th row contains the i-th rate as a function of the angle, i.e., r(theta).
    The matrix is then N*nangles.  
    """

    #Number of neurons
    N = len(v1_neurons)

    #Fill the rates
    rates = np.empty((N, nangles))
    for i in range(N):
        rates[i,:] = v1_activity[v1_activity["id"] == i]["rate"].values 

    return rates


#Get some masks that might 
def get_auxiliary_masks(v1_neurons):
    """
    Return two masks, is_in_l23 and is_tuned, which filter the indices for L2/3 and tuning conditions.
    """
    is_in_l23 = v1_neurons[v1_neurons["layer"] == "L2/3"]
    is_tuned  = v1_neurons[v1_neurons["tuning_type"] != "not_selective"]
    return is_in_l23, is_tuned

