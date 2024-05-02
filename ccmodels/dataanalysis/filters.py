import pandas as pd
import numpy as np

def filter_neurons(v1_neurons, layer=None, tuning=None, cell_type=None, proofread=None):
    """
    Convenience function. Filter neurons by several common characteristics at the same time.
    Leave parameters to None to not filter for them (default). Returns the table of the 
    neurons fulfilling the criteria.

    Parameters:

    v1_neurons : DataFrame
        neuron's properties DataFrame
    layer : string 
        The layer we want to filter for "L23" or "L4"
    tuning : string  
        whether if we want the neurons to be "tuned" or "untuned". One can get functionally matched neurons
        of any tuning using "matched", or neurons that were not matched by "unmatched". 
    cell_type : string
        "exc" or "inh" neurons 
    proofread : string
        filters for different levels of proofreading, labelled as "minimum" (axons are at least clean), 
        "decent" (axons have extended proofreading), "good" (axons are extended, dendrites are at least clean)
        and "perfect" (both are extended). Observe that "decent" is the minimum recommended by the Microns
        consortium to perform detailed analysis.
    """


    #All true, no filtering at all 
    nomask = np.ones(len(v1_neurons), dtype=bool) 

    #Get the filters for layer and cell if the user asked for them
    mask_layer = v1_neurons["layer"] == layer if layer != None else nomask 
    mask_cellt = v1_neurons["cell_type"] == cell_type if cell_type != None else nomask 

    #Get the filter for tuned/untuned neurons.
    #The same filter allows us to see if our neuron if functionally matched or not
    if tuning == None:
        mask_tuned = nomask
    elif tuning == "tuned":
        mask_tuned = (v1_neurons["tuning_type"] == "direction") | (v1_neurons["tuning_type"] == "orientation")
    elif tuning == "untuned":
        mask_tuned = v1_neurons["tuning_type"] == "not_selective"
    elif tuning == "matched":
        mask_tuned = v1_neurons["tuning_type"] != "not_matched"
    elif tuning == "unmatched":
        mask_tuned = v1_neurons["tuning_type"] == "not_matched"


    if proofread == None:
        mask_proof = nomask
    elif proofread == "minimum":
        mask_proof = v1_neurons["axon_proof"] != "non"
    elif proofread == "decent":
        mask_proof = v1_neurons["axon_proof"] == "extended"
    elif proofread == "good":
        mask_proof = (v1_neurons["axon_proof"] == "extended") & (v1_neurons["dendr_proof"] != "non")
    elif proofread == "perfect":
        mask_proof = (v1_neurons["axon_proof"] == "extended") & (v1_neurons["dendr_proof"] == "extended")

    
    return v1_neurons[mask_layer & mask_cellt & mask_tuned & mask_proof]


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

def filter_connections(v1_neurons, v1_connections, layer=None, tuned=None, cell_type=None, proofread, who="pre"):
    """
    Convenience function to call filter_neurons + synapses_by_id, i.e. filtering neurons by a criterium
    and then returning all connections fulfilling this condition. 
    Needs neuron table, connection table, and then filter by layer, tuned or cell_type (see filter_neurons) and 
    filtering pre/post or both neurons (see synapses by id).
    """

    neurons_filtered = filter_neurons(v1_neurons, layer, tuned, cell_type, proofread)
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


    