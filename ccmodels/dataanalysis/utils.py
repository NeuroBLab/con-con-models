import pandas as pd
import numpy as np
import ccmodels.dataanalysis.dfs2numpy as d2n

# ------------------------------------------------------------
# ------------------ LOAD DATA     --------------------------- 
# ------------------------------------------------------------

def load_data(half_angle=True, nangles=16, path="../con-con-models", version="343"):
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
    datasets = []

    if version=="343":
        v1_neurons = pd.read_csv(f'{path}/data/preprocessed/v1_neurons.csv')
        v1_connections = pd.read_csv(f'{path}/data/preprocessed/v1_connections.csv')
        rates_table = pd.read_csv(f'{path}/data/preprocessed/v1_activity.csv')
    elif version=="661":
        v1_neurons = pd.read_csv(f'{path}/data/preprocessed/unit_table.csv')
        v1_connections = pd.read_csv(f'{path}/data/preprocessed/connections_table.csv')
        rates_table = pd.read_csv(f'{path}/data/preprocessed/activity_table.csv')

        #Ensure all ids are from 0 to N-1, being N number of neurons. 
        #Rename id names.
        d2n.remap_all_tables(v1_neurons, v1_connections, rates_table)


    #If the half-angle thing is used, then we need to remap all of neurons's angles  
    if half_angle:
        v1_neurons.loc[:, "pref_ori"] = constrain_angles(v1_neurons["pref_ori"].values, nangles=8)

    #Once angles have been constrained, construct the delta ori values 
    v1_connections["delta_ori"] = construct_delta_ori(v1_neurons, v1_connections, orientation_only=half_angle)

    #If we want activity, read the table and return the Nx16 matrix directly
    if version=="343":
        rates = d2n.get_rates_matrix(v1_neurons, rates_table)
    elif version=="661":
        #Here we have inhibitory neurons too
        exc_neurons = filter_neurons(v1_neurons, cell_type="excitatory")
        rates = d2n.get_rates_matrix(exc_neurons, rates_table)


    #If we are working only with the orientation, assume that the rate we have for that 8 angles
    #is just the average between both
    if half_angle:
        rates = 0.5*(rates[:, 0:nangles//2] + rates[:, nangles//2:nangles])             

    return v1_neurons, v1_connections, rates




def construct_delta_ori(v1_neurons, v1_connections, orientation_only=True):
    """
    Given the tables of neurons and connections, get the array of delta orientations for each link and returns it.
    """

    dtheta = np.empty(len(v1_connections))

    for i, (id_pre, id_post) in enumerate(v1_connections[["pre_id", "post_id"]].values):
        dtheta[i] = angle_diff(v1_neurons.loc[id_post, "pref_ori"], v1_neurons.loc[id_pre, "pref_ori"], half=orientation_only) 

    return dtheta

# ------------------------------------------------------------
# ------------------ HANDY FILTERS --------------------------- 
# ------------------------------------------------------------

def filter_neurons(v1_neurons, layer=None, tuned=None, cell_type=None):
    """
    Convenience function. Filter neurons by several common characteristics at the same time.
    Leave parameters to None to not filter for them (default). Returns the table of the 
    neurons fulfilling the criteria.

    Parameters:

    v1_neurons : DataFrame
        neuron's properties DataFrame
    layer : string 
        The layer we want to filter for, L2/3 or L4
    tuned : bool 
        whether if we want the neurons to be tuned or untuned
    cell_type : string
        excitatory or inhibitory neurons 
    """


    #All true, produces no masking
    nomask = np.ones(len(v1_neurons), dtype=bool) 

    #Get the filters for layer and cell
    mask_layer = v1_neurons["layer"] == layer if layer != None else nomask 
    mask_cellt = v1_neurons["cell_type"] == cell_type if cell_type != None else nomask 

    #Get the filter for tuned/untuned neurons
    if tuned == None:
        mask_tuned = nomask
    elif tuned:
        mask_tuned = v1_neurons["tuning_type"] != "not_selective"
    else:
        mask_tuned = v1_neurons["tuning_type"] == "not_selective"
    
    return v1_neurons[mask_layer & mask_cellt & mask_tuned]


def synapses_by_id(neurons_id, v1_connections, who="pre"):
    """
    Given the ids of the neurons we want to filter for, grab the synapses that have ids matching for
    the pre- or post- synaptic neurons (or both).

    Parameters

    neurons_id : np.array
        Array with the IDs of the neurons we are filtering for
    v1_connections : DataFrame
        Dataframe with connectivity information
    who : string
        Can be "pre" (default), "post" or "both". If pre/post, selects pre/postsynaptic neurons which are contained in 
        the neurons_id array. If "both", it needs both IDs to be present.
    """

    if who=="pre":
        return v1_connections[v1_connections["pre_id"].isin(neurons_id)]
    elif who=="post":
        return v1_connections[v1_connections["post_id"].isin(neurons_id)]
    elif who=="both":
        return v1_connections[v1_connections["pre_id"].isin(neurons_id) & v1_connections["post_id"].isin(neurons_id)]

def filter_connections(v1_neurons, v1_connections, layer=None, tuned=None, cell_type=None, who="pre"):
    """
    Convenience function to call filter_neurons + synapses_by_id, i.e. filtering neurons by a criterium
    and then returning all connections fulfilling this condition. 
    Needs neuron table, connection table, and then filter by layer, tuned or cell_type (see filter_neurons) and 
    filtering pre/post or both neurons (see synapses by id).
    """

    neurons_filtered = filter_neurons(v1_neurons, layer, tuned, cell_type)
    return synapses_by_id(neurons_filtered["id"], v1_connections, who)

def connections_to(post_id, v1_connections, only_id=True):
    """
    Get the indices of the presynaptic neurons pointing to post_id
    """

    if only_id:
        return v1_connections[v1_connections["post_id"] == post_id]["pre_id"]
    else:
        return v1_connections[v1_connections["post_id"] == post_id]

def connections_from(pre_id, v1_connections):
    """
    Get the indices of the postsynaptic to which pre_id points  
    """
    return v1_connections[v1_connections["pre_id"] == pre_id]["post_id"]

# ---------------------------------------------------------------------------------
# ------------------ EXTRA INFO TO THE CONNECTION TABLE --------------------------- 
# ---------------------------------------------------------------------------------


def add_layerinfo_to_connections(v1_neurons, v1_connections, who="pre"):
    """
    Add columns to the v1_connections table in order to know the layer of pre and post synaptic neurons
    immediately. Set which ones to add by setting the argument who="pre", "post" or "both".
    """

    #Avoid modifying the original, by copying
    v1_conn_withlayer = v1_connections.copy() 

    #The same thing for pre and post: add the respective columns...
    if who == "pre" or who == "both":
        #Create a new column and fill it with a default value
        v1_conn_withlayer["pre_layer"] = "L2/3"

        #Then, search all neurons in L4 and set them in the new table
        #l4_neurons_ids = filter_neurons(v1_neurons, layer="L4")["id"]
        #mask_l4 = v1_conn_withlayer["pre_id"].isin(l4_neurons_ids)

        #Then, obtained the indices of all connections with presynaptic neurons in L4 and set them in the new table
        mask_l4 = filter_connections(v1_neurons, v1_connections, layer="L4").index.values
        v1_conn_withlayer.loc[mask_l4, "pre_layer"] = "L4"

    if who == "post" or who == "both":
        #By definition, the way we constructed the data, no postsynaptic neuron is in L4, so just fill it up!!
        v1_conn_withlayer["post_layer"] = "L2/3"

    
    #Return result
    return v1_conn_withlayer


def tuning_encoder(v1_neurons, v1_connections):
    '''
    Takes the neurons and synapses properties. Returns a table with two columns, pre_tuned and post_tuned, 
    which are True when the corresponding neuron is tuned. 

    Returns: 
    Updated v1_connections including new columns pre_tuned and post_tuned for the tuning of 
    the pre and post synaptic neurons, respectively
    '''

    #Get the neurons which are tuned
    tuned_neurons_ids = filter_neurons(v1_neurons, tuned=True)["id"]

    #Initialize a new table with the tuning set to false by default
    v1_conn_withtuning = v1_connections.copy()
    v1_conn_withtuning["pre_tuned"] = False
    v1_conn_withtuning["post_tuned"] = False

    #Look which IDs of pre and post synaptic neurons are inside of the tuned IDs we got, and set those to true
    v1_conn_withtuning.loc[v1_conn_withtuning["pre_id"].isin(tuned_neurons_ids), "pre_tuned"] = True
    v1_conn_withtuning.loc[v1_conn_withtuning["post_id"].isin(tuned_neurons_ids), "post_tuned"] = True

    return v1_conn_withtuning




# ------------------------------------------------------------
# ----------------------- TUNING HELPERS ---------------------
# ------------------------------------------------------------


def split_by_tuning(v1_connections):
    """
    Utility function that generates a single Pandas dataframe for each combination of tuned-untuned connectivity by layer,
    i.e., returns a table containing only L2/3 TUNED to L2/3 TUNED, L4 UNTUNED to L2/3 TUNED, and so on...

    Parameters:
    v1_connections: the list of connections between neurons. Must include information about the layer of the
    presynaptic neurons, which can be obtained by add_layerinfo_to_connections
    c_tuning: generated by tuning_encoder, a table indicating the tuning of each synapse
    """

    tables = {}

    #Are the pre and post neurons tuned or not?
    tuned_tuned = v1_connections['pre_tuned']   &  v1_connections['post_tuned']  
    tuned_untun = v1_connections['pre_tuned']   &  ~v1_connections['post_tuned'] 
    untun_tuned = ~v1_connections['pre_tuned']  &  v1_connections['post_tuned']  
    untun_untun = ~v1_connections['pre_tuned']  &  ~v1_connections['post_tuned']  

    #Which ones is the presynaptic thing coming from?
    l4  = v1_connections['pre_layer'] == 'L4'
    l23 = v1_connections['pre_layer'] == 'L2/3'


    #L4 -> L2/3
    tables["l4t_l23t"] =  v1_connections[tuned_tuned & l4]
    tables["l4t_l23u"] =  v1_connections[tuned_untun & l4]
    tables["l4u_l23t"] =  v1_connections[untun_tuned & l4]
    tables["l4u_l23u"] =  v1_connections[untun_untun & l4]

    #L2/3 -> L2/3      
    tables["l23t_l23t"] =  v1_connections[tuned_tuned & l23]
    tables["l23t_l23u"] =  v1_connections[tuned_untun & l23]
    tables["l23u_l23t"] =  v1_connections[untun_tuned & l23]
    tables["l23u_l23u"] =  v1_connections[untun_untun & l23]

    return tables 



# ------------------------------------------------------------
# ---------------------- WORKING WITH ANGLES -----------------
# ------------------------------------------------------------

def constrain_angles(thetas, nangles=16, negatives=True):
    """
    Constrain the angle indices to be in [0, nangles], which is
    sometimes necessary to operate
    """
    new_thetas = thetas.copy()
    
    #Negatives becomes 16 - X
    if negatives:
        negative = thetas< 0
        new_thetas[negative] = nangles + thetas[negative] #We put a + because they are already negative

    #Large ones bounded in [0, 16]
    large = np.abs(thetas) >= nangles 
    new_thetas[large] = np.sign(thetas[large]) * (thetas[large] % nangles) #Python modulo always return positive, so add the sign manually 

    return new_thetas

def angle_diff(pre, post, nangles=16, half=True):
    """
    Computes a signed difference between pre a post, by taking into account periodic boundaries.
    In this way, we get differences in [-k, ..., 0, ...k], being nangle-k mapped to -k until -nangle//2,
    where results jump to be positive. 
    """
    d = post - pre
    max_angle = nangles//4 if half else nangles//2

    if d <= -max_angle:
        return d + 2*max_angle
    elif d > max_angle:
        return d - 2*max_angle
    else:
        return d

def angle_dist(pre, post, nangles=16, half=True):
    """
    Classic distance with boundary conditions between angles pre and post, given as integers.
    """
    d = abs(post - pre)
    max_angle = nangles//4 if half else nangles//2
    return min(d, max_angle - d)







# ------------------------------------------------------------
# ---------------------- ACTIVITY HELPER -----------------
# ------------------------------------------------------------

def shift_rates(v1_neurons, pre_ids, post_id, rates):
    """
    Shift all the rates corresponding to the pre_ids, so that the post_id neuron
    would be oriented at angle = 0.

    Parameters
    v1_neurons : DataFrame
        The table with info of all the neurons.
    pre_ids : arraylike
        An array with all the ids of the considered presynaptic neurons 
    post_id : int
        The selected postsynaptic neuron id
    rates : numpy matrix
        The matrix with the rates information
    """
    if isinstance(post_id, (int, float, np.int64, np.float64)): 
        rates_selected = rates[pre_ids, :]
        post_pref_ori  = v1_neurons.loc[post_id, "pref_ori"]
        return  np.roll(rates_selected, -post_pref_ori, axis=1) 
    else:
        raise ValueError("shift_rates accepts only a scalar (int) postsynaptic id to work.")

def shift_multi(rates, rollamount):
    """
    Perform a roll for each one of the rows of the vector rates by
    the quantities defined in rollamount

    Parameters
    ==========
    rates : numpy array NxM
        An array to be reshuffled 
    rollamount : numpy array N
        rollamount[i] says how much to roll rates[i, :]
    """

    return np.array([np.roll(rates[i, :], -r) for i,r in enumerate(rollamount)])



def shuffle_neurons(ids, rates):
    """
    Shuffle the rates of the selected neurons, taken by their id 
    """
    #Shuffle the selected columns (inplace) 
    for id in ids:
        np.random.shuffle(rates[id, :])

    return 


def get_untuned_rate(v1_neurons, rates):
    """
    Returns a modified rate matrix where the rows corresponding to untuned neurons
    are set to their average 
    """

    #Find the untuned neurons
    untuned_ids = filter_neurons(v1_neurons, tuned=False)["id"]
    rates_unt = rates.copy()
    #Substitute the not tuned ones with the mean rate accross all its angles
    #The newaxis thing allows it to be assigned doing rates[ids, :] = directly
    rates_unt[untuned_ids, :] = np.mean(rates[untuned_ids, :], axis=1)[:, np.newaxis]   
    return rates_unt


