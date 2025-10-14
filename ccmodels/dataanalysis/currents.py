import numpy as np
import pandas as pd


import ccmodels.dataanalysis.utils as utl
import ccmodels.dataanalysis.filters as fl
import ccmodels.utils.angleutils as au




# ===================================================
# ------- INPUT CURRENT AUXILIARY FUNCTIONS ---------
# ===================================================

def get_input_to_neuron(v1_neurons, v1_connections, post_id, vij, rates, shifted=True):
    """
    Get the input to a single postsynaptic neuron, allowing to shift presynaptic inputs too.

    Parameters
    vij : NxN numpy matrix
        Adjacency matrix
    rates : Nxnangles numpy matrix
        Contains the respose to each estimuli for each neuron
    post_id : 
        Index of the selected postsynaptic neurons 

    Returns 
    post_input : numpy array
        Input to the selected postsynaptic neurons, as a Numpy array
    """

    pre_ids = fl.connections_to(post_id, v1_connections)

    #get_currents_subset returns [[current]], so select the first element. Also post_id need to be an array
    return get_currents_subset(v1_neurons, vij, rates, post_ids=[post_id], pre_ids=pre_ids, shift=shifted)[0]

def get_currents_subset(v1_neurons, vij, rates, post_ids=None, pre_ids=None, shift=False):
    """
    Get the input from the selected presynaptic neurons to the selected postsynaptic neurons.

    Parameters
    vij : NxN numpy matrix
        Adjacency matrix
    rates : Nxnangles numpy matrix
        Contains the respose to each estimuli for each neuron
    post_ids : numpy array
        Indices of the selected postsynaptic neurons. IF None (default) just use all the
        postsynaptic neurons
    pre_ids : numpy array
        Indices of the selected presynaptic neurons. If None (default), just return all the presynaptic
        input to the selected neurons.
    shift : bool
        Indicates whether the results should be shifted or not (False by default)

    Returns 
    post_input : numpy array
        Input to the selected postsynaptic neurons, as a Numpy array
    """

    #If none, then just return the entire multiplication to get all currents 
    if pre_ids is None and post_ids is None:
        inputcur = np.matmul(vij, rates)
        post_ids = np.arange(rates.size)
    #If only one is none, then get the correct subset
    elif pre_ids is None:
        inputcur = np.matmul(vij[post_ids, :], rates)
    elif post_ids is None:
        inputcur = np.matmul(vij[:, pre_ids], rates[pre_ids, :])
    #Use np.ix to select by row and column directly
    else:
        inputcur = np.matmul(vij[np.ix_(post_ids, pre_ids)], rates[pre_ids, :])

    #Shift currents when indicated, so the input current has angle 0 at the preferred postsynaptic
    #Each current is shifted to a different value, the one of its post neuron, via shift_multi 
    if shift:
        if post_ids is None: 
            inputcur = utl.shift_multi(inputcur, v1_neurons["pref_ori"])
        else:
            inputcur = utl.shift_multi(inputcur, v1_neurons.loc[post_ids, "pref_ori"])

    return inputcur


def get_input_virtual_presynaptic(v1_neurons, selected_connections, rates):
    """
    Assume we select several synapses, and assume that all their presynaptic neurons are 
    pointing to a virtual postsynaptic neuron. Get the input current of the virtual neuron.
    Inputs will be always shifted to their postsynaptic neurons so the virtual neuron has pref_ori = 0

    Parameters:
    selected_connections: table of connections, consisting in all selected (pre) synapses
    rates: vector containing rates for the whole system
    """

    #Get the indices and the synapses' weight
    pre_ids = selected_connections["pre_id"]
    post_ids = selected_connections["post_id"]
    volumes = selected_connections["syn_volume"].values

    #Just shift every presynaptic rate by each postsynaptic neuron's pref ori 
    angle_post = v1_neurons.loc[post_ids, "pref_ori"]
    shifted_rates = utl.shift_multi(rates[pre_ids], angle_post)

    #Compute the result to thie virtual neuron
    return np.dot(volumes, shifted_rates) 

def get_current_normalization(v1_neurons, vij, rates):
    """
    Normalization of the currents is the mean presynaptic current
    """
    all_currents = get_currents_subset(v1_neurons, vij, rates)
    return all_currents.mean()



# ===================================================
# --------- COMPUTATION OF OBSERVABLES --------------
# ===================================================



def single_synapse_current(v1_neurons, v1_connections, vij, rates, shifted=True):
    """
    Compute real input from the data using just a single neuron. 
    """

    #Find pairs of tuned neurons, with presynaptic ones coming from layer L2/3
    tuned_outputs = fl.filter_connections(v1_neurons, v1_connections, tuning="tuned", who="both")
    tuned_outputs = fl.filter_connections(v1_neurons, tuned_outputs, layer="L23", who="pre")

    #Find tuned neurons in L2/3 
    tuned_neurons = fl.filter_neurons(v1_neurons, layer="L23", tuning="tuned") 

    #Select a random postsynaptic neuron and its presynaptic ones 
    selected_neuron = tuned_neurons.sample(1)["id"].values[0]

    #Then just get the currents to this neuron using only the tuned outputs
    return get_input_to_neuron(v1_neurons, tuned_outputs, selected_neuron, vij, rates, shifted=shifted)



def compute_distrib_diffrate_allsynapses(v1_neurons, v1_connections, vij, rates, shifted=True, nangles=16, half=True):
    """
    Get the actual synaptic currents from L2/3 and L4. Shift all them to postsynaptic angle = 0 
    and then compute the difference between I(0)-I(pi/2). Return this difference for each computed current.
    """

    #Indices to measure the difference
    index_zero = 0 
    if half:
        index_pihalf = nangles//8 
    else:
        index_pihalf = nangles//4 

    #Find pairs of tuned neurons, with presynaptic ones coming from specific layer 
    l23_conns = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=["L23", "L23"], tuning=["tuned", "matched"])
    l4_conns = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=["L4", "L23"], tuning=["tuned", "matched"])

    #Substitute the untuned neurons with the average rate
    rates_untuned = utl.get_untuned_rate(v1_neurons, rates) 

    #Get number of angles, and set up a Dataframe based on it, that we will fill
    #Keep each result in a separate table depending on layer of presynaptic
    diffs = {"L23" : [], "L4"   : []} 

    #Compute the currents
    #maxcurr = get_current_normalization(v1_neurons, vij, rates_untuned)
    curr_l23 = get_currents_subset(v1_neurons, vij, rates_untuned, pre_ids=l23_conns["pre_id"], post_ids=l23_conns["post_id"], shift=shifted)
    curr_l4  = get_currents_subset(v1_neurons, vij, rates_untuned, pre_ids=l4_conns["pre_id"], post_ids=l4_conns["post_id"], shift=shifted)

    maxcurr = max(curr_l23.max(), curr_l4.max())

    #Normalize currents
    curr_l23 /= maxcurr
    curr_l4  /= maxcurr

    #Compute the inputs for all the selected neurons. Do it separately for each layer
    for layer, currlayer in zip(["L23", "L4"], [curr_l23, curr_l4]):
        #Difference between pi/half and 0
        for current in currlayer: 
            diffs[layer].append(current[index_zero] - current[index_pihalf]) 

    return diffs

def sample_prefori(v1_neurons, tuned_connections, nexperiments, rates, nsamples):
    """
    Assume a virtual presynaptic neuron. Sample connections to it and compute the current. Determine the 
    preferred orientation from the current maximum. Repeat for nexperiments, and compute the probability
    of each preferred orientation. 
    """

    #The number of columns of the rates variables give the number of angles
    nangles = rates.shape[1]

    #Get the untuned rates
    rates_untuned = utl.get_untuned_rate(v1_neurons, rates)

    #Prepare to do experiments...
    prob_pref_ori = {} 
    currents = {}
    prob_pref_ori['Total'] = np.zeros(nangles) 
    prob_pref_ori['L23'] = np.zeros(nangles) 
    prob_pref_ori['L4'] = np.zeros(nangles) 

    currents['Total'] = np.zeros(nangles) 
    currents['L23'] = np.zeros(nangles) 
    currents['L4'] = np.zeros(nangles) 

    for i in range(nexperiments): 
        #Sample a bunch of neurons
        neuron_sample = tuned_connections.sample(nsamples, replace=True) 

        #Compute the current they get 
        current = get_input_virtual_presynaptic(v1_neurons, neuron_sample, rates_untuned) 

        #Use the current to determine the preferred orientation
        idx_prefrd_ori = np.argmax(current)
        prob_pref_ori['Total'][idx_prefrd_ori] += 1

        #Now we get the components from L23 and L4. Filter the neurons that were in L23...
        neuron_sample_L23 = fl.filter_connections(v1_neurons, neuron_sample, layer='L23', who='pre')
        current_L23 = get_input_virtual_presynaptic(v1_neurons, neuron_sample_L23, rates_untuned) 
        #The L4 current is the difference between total and L@3
        neuron_sample_L23 = fl.filter_connections(v1_neurons, neuron_sample, layer='L4', who='pre')
        #current_L4  = get_input_virtual_presynaptic(v1_neurons, neuron_sample_L23, rates_untuned) 
        current_L4 = current - current_L23

        #Preferred location for each component of the currents
        idx_prefrd_ori = np.argmax(current_L23)
        prob_pref_ori['L23'][idx_prefrd_ori] += 1
        idx_prefrd_ori = np.argmax(current_L4)
        prob_pref_ori['L4'][idx_prefrd_ori] += 1

        currents["Total"] += current
        currents["L23"] += current_L23
        currents["L4"] += current_L4
    
    #Normalize with respect to all experiments 
    for layer in ['Total', 'L23', 'L4']:
        prob_pref_ori[layer] /= nexperiments 
        currents[layer] /= nexperiments
        prob_pref_ori[layer + "_error"] = np.sqrt((prob_pref_ori[layer] * (1 - prob_pref_ori[layer])) / nexperiments) 

    #Return probabilities and the first two moments of the bootstrap distribution  
    return prob_pref_ori, currents

def fraction_prefori_predicted(v1_neurons, tuned_connections, vij, rates):
    """
    Computes the fraction of neurons for which the preferred orientation is predicted by the currents
    """

    all_pref_oris = v1_neurons['pref_ori'].values

    currents = np.matmul(vij, rates)
    idx_pref_ori = np.argmax(currents, axis=1)

    return au.signed_angle_dist_vectorized(idx_pref_ori, all_pref_oris)

def bootstrap_mean_current(kee, v1_neurons, tuned_connections, rates, nexperiments):
    """
    Computes the average input current to neurons in the L2/3, as well as the proportion of it
    arriving from recurrent interactions and L4
    """

    #The number of columns of the rates variables give the number of angles
    nangles = rates.shape[1]

    #Get the untuned rates
    rates_untuned = utl.get_untuned_rate(v1_neurons, rates)

    #Ids of the L23/4 neurons, in order to be able to filter the corresponding pre/postsynaptic neurons
    post_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="tuned")["id"]
    l23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="matched", proofread="minimum")["id"]
    l4_ids  = fl.filter_neurons(v1_neurons, layer="L4",  tuning="matched", proofread="minimum")["id"]

    layers = ["Total", "L23", "L4"]

    #Initialize the currents 
    avr_cur = {}
    std_cur = {}
    for key in layers: 
        avr_cur[key] = np.zeros(nangles)
        std_cur[key] = np.zeros(nangles)

    #Prepare to do experiments...
    for i in range(nexperiments): 
        #Bootstrap sample the entire table. If only one experiment is used, we do not bootstrap, just use the 
        #table as it is.
        sample_syn = tuned_connections.sample(n=kee, replace=nexperiments!=1) 

        #In this bootstrap, get the connections from the presynaptic layer to L23
        con_boots_total = fl.synapses_by_id(sample_syn, post_ids=post_ids, who="post")
        con_boots_l23   = fl.synapses_by_id(sample_syn, pre_ids=l23_ids, post_ids=post_ids, who="both")
        con_boots_l4    = fl.synapses_by_id(sample_syn, pre_ids=l4_ids, post_ids=post_ids, who="both")

        #Compute the currents we get from those 
        current = {}
        current["Total"] = get_input_virtual_presynaptic(v1_neurons, con_boots_total, rates_untuned) 
        current["L23"]   = get_input_virtual_presynaptic(v1_neurons, con_boots_l23,   rates_untuned) 
        current["L4"]    = get_input_virtual_presynaptic(v1_neurons, con_boots_l4,    rates_untuned) 

        #Get the average and error of the current for each angle
        for key in layers: 
            avr_cur[key] += current[key]
            std_cur[key] += current[key]**2 

    #Finish averages
    for key in layers: 
        avr_cur[key] /= nexperiments
        std_cur[key] /= nexperiments
        std_cur[key] = np.sqrt((std_cur[key] - avr_cur[key]**2) / nexperiments)

    #Normalize accordingly
    maxcur = np.max(avr_cur["Total"])
    for key in layers: 
        avr_cur[key] /= maxcur
        std_cur[key]/= maxcur

    #Return first two moments of the bootstrap average estimator. 
    return avr_cur, std_cur 

import ccmodels.dataanalysis.processedloader as loader

def bootstrap_system_currents(v1_neurons, tuned_connections, rates, nexperiments, frac=1.0, replace=True, shift=True):
    """
    Computes the average input current to neurons in the L2/3, as well as the proportion of it
    arriving from recurrent interactions and L4
    """

    #The number of columns of the rates variables give the number of angles
    nangles = rates.shape[1]

    #In this case, we process the entire table
    if nexperiments==1 and frac==1.0:
        replace=False

    #Get the untuned rates
    #rates_untuned = utl.get_untuned_rate(v1_neurons, rates)

    #Ids of the L23/4 neurons, in order to be able to filter the corresponding pre/postsynaptic neurons
    l23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="tuned")["id"]

    prel23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="matched", proofread='minimum')["id"]
    prel4_ids  = fl.filter_neurons(v1_neurons, layer="L4",  tuning="matched", proofread='minimum')["id"]
    preallids = fl.filter_neurons(v1_neurons, tuning="matched", proofread='minimum')["id"]

    layers = ["Total", "L23", "L4"]

    #Initialize the currents 
    avr_cur = {}
    std_cur = {}
    for key in layers: 
        avr_cur[key] = np.zeros((nangles, nexperiments))
        std_cur[key] = np.zeros((nangles, nexperiments))

    rates = utl.get_untuned_rate(v1_neurons, rates)

    #Prepare to do experiments...
    for i in range(nexperiments): 
        #Bootstrap sample the entire table. If only one experiment is used, we do not bootstrap, just use the 
        #table as it is.
        con_boots_total = tuned_connections.sample(frac=frac, replace=replace) 
        vij = loader.get_adjacency_matrix(v1_neurons, con_boots_total)

        #Compute the currents we get from those 
        current = {}
        current["Total"] = get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids, pre_ids=preallids, shift=shift)
        current["L23"] = get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids, pre_ids=prel23_ids, shift=shift)
        current["L4"] = get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids, pre_ids=prel4_ids, shift=shift)

        #Get the average and error of the current for each angle
        for key in layers: 
            avr_cur[key][:, i] = np.sum(current[key], axis=0) 
            std_cur[key][:, i] = np.std(current[key], axis=0) 


    #Normalize accordingly
    maxcur = 1. #np.max(avr_cur["Total"])
    for key in layers: 
        avr_cur[key] /= maxcur 
        std_cur[key] /= maxcur

    #Return first two moments of the bootstrap average estimator. 
    return avr_cur, std_cur 

def bootstrap_system_currents_shuffle(v1_neurons, tuned_connections, rates, nexperiments, frac=1.0, replace=True, shift=True):

    #The number of columns of the rates variables give the number of angles
    nangles = rates.shape[1]

    #In this case, we process the entire table
    if nexperiments==1 and frac==1.0:
        replace=False

    #Get the untuned rates
    #rates_untuned = utl.get_untuned_rate(v1_neurons, rates)

    #Ids of the L23/4 neurons, in order to be able to filter the corresponding pre/postsynaptic neurons
    l23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="tuned")["id"]

    prel23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="matched", proofread='minimum')["id"]
    prel4_ids  = fl.filter_neurons(v1_neurons, layer="L4",  tuning="matched", proofread='minimum')["id"]
    preallids = fl.filter_neurons(v1_neurons, tuning="matched", proofread='minimum')["id"]
    #prel23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="matched")["id"]
    #prel4_ids  = fl.filter_neurons(v1_neurons, layer="L4",  tuning="matched")["id"]
    #preallids = fl.filter_neurons(v1_neurons, tuning="matched")["id"]

    layers = ["Total", "L23", "L4"]

    #Initialize the currents 
    avr_cur = {}
    for key in layers: 
        avr_cur[key] = np.zeros((nangles, nexperiments))

    tuned = fl.filter_neurons(v1_neurons, tuning='tuned')
    rates_shifted = pd.DataFrame(utl.shift_multi(rates[tuned['id'], :], tuned['pref_ori']))
    #rates_shifted = pd.DataFrame(utl.shift_multi(rates, v1_neurons['pref_ori']))

    #Prepare to do experiments...
    for i in range(nexperiments): 
        #Bootstrap sample the entire table. 
        con_boots_total = tuned_connections.sample(frac=frac, replace=replace) 

        #Bootsrap weights and rates
        con_boots_total.loc[:, 'syn_volume'] = tuned_connections.loc[:, 'syn_volume'].sample(frac=frac, replace=replace).values
        rates_sample = rates_shifted.sample(n=rates.shape[0], replace=replace)
        rates_sample = utl.shift_multi(rates_sample.values, -v1_neurons['pref_ori'])
        #rates_sample = rates_shifted.values
        #rates_sample = utl.shift_multi(rates_sample, v1_neurons['pref_ori'])

        #Set the untuned neurons
        rates_sample = utl.get_untuned_rate(v1_neurons, rates_sample)

        vij = loader.get_adjacency_matrix(v1_neurons, con_boots_total)

        #Compute the currents we get from those 
        current = {}
        current["Total"] = get_currents_subset(v1_neurons, vij, rates_sample, post_ids=l23_ids, pre_ids=preallids, shift=shift)
        current["L23"] = get_currents_subset(v1_neurons, vij, rates_sample, post_ids=l23_ids, pre_ids=prel23_ids, shift=shift)
        current["L4"] = get_currents_subset(v1_neurons, vij, rates_sample, post_ids=l23_ids, pre_ids=prel4_ids, shift=shift)

        #Get the average and error of the current for each angle
        for key in layers: 
            avr_cur[key][:, i] = np.sum(current[key], axis=0) 


    #Normalize accordingly
    maxcur = 1. #np.max(avr_cur["Total"])
    for key in layers: 
        avr_cur[key] /= maxcur 

    #Return first two moments of the bootstrap average estimator. 
    return avr_cur









def bootstrap_system_currents_peaks(v1_neurons, tuned_connections, rates, frac=1.0, shift=True, nexperiments = 100):
    """
    Computes the average input current to neurons in the L2/3, as well as the proportion of it
    arriving from recurrent interactions and L4
    """

    #Ids of the L23/4 neurons, in order to be able to filter the corresponding pre/postsynaptic neurons
    l23_ids_post = fl.filter_neurons(v1_neurons, layer="L23", tuning="tuned")["id"]
    l23_ids = fl.filter_neurons(v1_neurons, layer="L23", tuning="matched", proofread='minimum')["id"]
    l4_ids  = fl.filter_neurons(v1_neurons, layer="L4",  tuning="matched", proofread='minimum')["id"]
    allids = fl.filter_neurons(v1_neurons, tuning="matched", proofread='minimum')["id"]

    result = {}
    result['Total'] = np.zeros(8)
    result['L23'] = np.zeros(8)
    result['L4'] = np.zeros(8)

    #Prepare to do experiments...
    #Bootstrap sample the entire table. 
    for i in range(nexperiments):
        con_boots_total = tuned_connections.sample(frac=frac, replace=False) 

        vij = loader.get_adjacency_matrix(v1_neurons, con_boots_total)

        #Compute the currents we get from those 
        #current = {}
        #current["Total"] = get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids_post, pre_ids=allids, shift=shift)
        #current["L23"] = get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids_post, pre_ids=l23_ids, shift=shift)
        #current["L4"] = get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids_post, pre_ids=l4_ids, shift=shift)

        result["Total"] += get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids_post, pre_ids=allids, shift=shift)
        result["L23"]   += get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids_post, pre_ids=l23_ids, shift=shift)
        result["L4"]    += get_currents_subset(v1_neurons, vij, rates, post_ids=l23_ids_post, pre_ids=l4_ids, shift=shift)
    
    return result / nexperiments