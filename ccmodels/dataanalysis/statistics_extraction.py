import numpy as np
import pandas as pd

import ccmodels.dataanalysis.utils as utl
import ccmodels.dataanalysis.filters as fl 
import ccmodels.dataanalysis.processedloader as loader
import ccmodels.utils.angleutils as au


#TODO needs to be merged with new version of the preprocssing results
def get_number_connections(data_location = '../con-con-models/data', layer234_only=True):
    '''Reads in appropriate data and prepares it for plotting by calculating counts'''

    #TODO fill function with new table data format
    pass 
    """
    nonproof_inputs_sample = pd.read_csv(f'{data_location}/nonproof_inputs_sample.csv')
    proof_inputs_sample = pd.read_csv(f'{data_location}/proofread_inputs_sample.csv')

    nonproof_outputs_sample = pd.read_csv(f'{data_location}/nonproof_outputs_sample.csv')
    proof_outputs_sample = pd.read_csv(f'{data_location}/proofread_outputs_sample.csv')

    #These tables contain more layers. If we want to check only connectivity from layer 2,3,4, we do it in this way
    if layer234_only:
        mask = (nonproof_inputs_sample["cortex_layer"] == "L2/3") | (nonproof_inputs_sample["cortex_layer"] == "L4")
        nonproof_inputs_sample = nonproof_inputs_sample[mask]
        mask = (proof_inputs_sample["cortex_layer"] == "L2/3") | (proof_inputs_sample["cortex_layer"] == "L4")
        proof_inputs_sample = proof_inputs_sample[mask]
        mask = (nonproof_outputs_sample["cortex_layer"] == "L2/3") | (nonproof_inputs_sample["cortex_layer"] == "L4")
        nonproof_outputs_sample = nonproof_outputs_sample[mask]
        mask = (proof_outputs_sample["cortex_layer"] == "L2/3") | (proof_inputs_sample["cortex_layer"] == "L4")
        proof_outputs_sample = proof_outputs_sample[mask]

    #Count number of input neurons for each post synaptic neuron
    nonproof_inputs_counts = nonproof_inputs_sample.groupby('post_pt_root_id').count().reset_index()
    proof_inputs_counts = proof_inputs_sample.groupby('post_pt_root_id').count().reset_index()

    #Count number of output neurons for each pre synaptic neuron
    nonproof_outputs_counts = nonproof_outputs_sample.groupby('pre_pt_root_id').count().reset_index()
    proof_outputs_counts = proof_outputs_sample.groupby('pre_pt_root_id').count().reset_index()

    return nonproof_inputs_counts["id"], proof_inputs_counts["id"], nonproof_outputs_counts["id"], proof_outputs_counts["id"]
    """

def get_propotion_connections(data_location = '../con-con-models/data'):
    '''
    Performs a bootstrap analysis on the proportion of inputs from L2/3 and L4 neurons that are received by
    L2/3 neurons that are proofread VS that are not proofread
    '''

    #TODO fill function with new table data format
    pass

    """
    #Read data from layer L2/3
    proof_inputs_l23sample = pd.read_csv(f'{data_location}/proof_inputs_l23sample.csv')
    nonproof_inputs_l23sample = pd.read_csv(f'{data_location}/nonproof_inputs_l23sample.csv')


    #Group inputs by postsynaptic neurons and by layer
    layer_groups_nop = proof_inputs_l23sample.groupby(['post_pt_root_id','cortex_layer']).count()
    layer_groups_p = nonproof_inputs_l23sample.groupby(['post_pt_root_id','cortex_layer']).count()

    #Clean the dataframes
    #Rename the id column to be called n_connections and contain the number of postsynaptic units
    #Proofread
    layer_groups_p = layer_groups_p.reset_index().loc[:, ['post_pt_root_id', 'cortex_layer', 'id']].rename(columns = {'id':'n_connections'})
    #NON-proofread
    layer_groups_nop = layer_groups_nop.reset_index().loc[:, ['post_pt_root_id', 'cortex_layer', 'id']].rename(columns = {'id':'n_connections'})

    boots_propl_proof = bootstrap_layerinput_proportions(layer_groups_p, 'cortex_layer', 'n_connections', 
                                                         layer_labels = ['L2/3', 'L4'], n_iters = 100)
    
    boots_propl_noproof = bootstrap_layerinput_proportions(layer_groups_nop, 'cortex_layer', 'n_connections', 
                                                           layer_labels = ['L2/3', 'L4'], n_iters = 100)
    
    
    return boots_propl_proof, boots_propl_noproof
    """


def prob_conectivity_tuned_untuned(v1_neurons, v1_connections, nsamples=100):
    """
    Perform a number nsamples of bootstrap samples of the connection probability, and returns a matrix
    of probability of connection between tuned/untuned populations of layers 2/3 and 4 to layer 2/3.
    """

    #Connection probability between combinations of tuned and untuned inputs/outputs
    #Rows: L2/3T, L2/3U (post synaptic)
    #Columns: L2/3T, L2/3U,  L4T, L4U (pre synaptic)
    sampled_probabilities = {"l4t_l23t": 0.0, "l4t_l23u": 0.0, "l4u_l23t": 0.0, "l4u_l23u": 0.0, 
                             "l23t_l23t": 0.0, "l23t_l23u": 0.0, "l23u_l23t": 0.0, "l23u_l23u": 0.0,} 

    #Get the tuning for each synapse and add it to the table.
    #We add it instead of keeping because for the sampling it is simpler & faster to have them in 
    #the columns already.
    connections_encoded = utl.tuning_encoder(v1_neurons, v1_connections)

    #Sampl from the table
    for i in range(nsamples):

        #Bootstrap a number of connections with repetitions
        v1_connections_samp = connections_encoded.sample(len(connections_encoded), replace = True)

        #Separate the samples in both tuned/untuned and layer.
        tuning_tables = utl.split_by_tuning(v1_connections_samp)


        #Normalising constants, total number of connections, both for L4
        layer_4_tuned_out = len(v1_connections_samp[v1_connections_samp['post_tuned'] 
                                                & (v1_connections_samp["pre_layer"]=="L4") ])

        layer_4_untuned_out = len(v1_connections_samp[~v1_connections_samp['post_tuned'] 
                                                  & (v1_connections_samp["pre_layer"]=="L4")])

        #And for L2/3
        layer_23_tuned_out = len(v1_connections_samp[v1_connections_samp['post_tuned'] 
                                                 & (v1_connections_samp["pre_layer"]=="L23")])


        layer_23_untuned_out = len(v1_connections_samp[~v1_connections_samp['post_tuned']
                                                & (v1_connections_samp["pre_layer"]=="L23")])

        #Get the normalizing constants in a handy dictionary so we can compute the 
        #probability of "how much we have in each tuning_table" divided by the total number of links
        norm = {"l4t_l23t": layer_4_tuned_out, "l4t_l23u": layer_4_untuned_out, 
                "l4u_l23t": layer_4_tuned_out, "l4u_l23u": layer_4_untuned_out, 
                "l23t_l23t": layer_23_tuned_out, "l23t_l23u": layer_23_untuned_out, 
                "l23u_l23t": layer_23_tuned_out, "l23u_l23u": layer_23_untuned_out} 

        #Add to the average. We divide the amount of connections in the table by the total
        for combination in sampled_probabilities:
            sampled_probabilities[combination] += len(tuning_tables[combination]) / norm[combination]

    #Finish average
    for key in sampled_probabilities:
        sampled_probabilities[key] /= nsamples

    #Finish average and return
    return sampled_probabilities



def strength_tuned_untuned(v1_neurons, v1_connections):
    """
    Returns the matrix of connection strength from layer 2/3 and 4 to layer 2/3,
    normalized to the strenght of recurrent layer 2/3 connections.
    """


    sampled_strenghts = {"l4t_l23t": 0.0, "l4t_l23u": 0.0, "l4u_l23t": 0.0, "l4u_l23u": 0.0, 
                             "l23t_l23t": 0.0, "l23t_l23u": 0.0, "l23u_l23t": 0.0, "l23u_l23u": 0.0,} 

    #Extract tuning combinations. Add presynaptic layer info to the synapses  
    #l4_combinations, l23_combinations = tuning_segmenter(v1_connections)
    v1_connections = utl.tuning_encoder(v1_neurons, v1_connections)
    tuning_tables = utl.split_by_tuning(v1_connections)

    #Neurons in layer 2/3...
    #l23_neurons_id = get_neurons_in_layer(v1_neurons, "L23")["id"]
    l23_neurons_id = fl.filter_neurons(v1_neurons, layer="L23")["id"] 

    #Normalization by the mean size of layer L2/3
    normcstr = np.mean(v1_connections.loc[v1_connections["pre_layer"] == "L23", 'syn_volume'])

    #Fill our matrix accordingly. See also the previous function
    for combination in sampled_strenghts:
        sampled_strenghts[combination] = np.mean(tuning_tables[combination]['syn_volume']) / normcstr

    return sampled_strenghts


def bootstrap_prob_tuned2tuned(v1_neurons, v1_connections, pre_layer, proofread=["ax_clean", None], half=True, nangles=16, n_samps=1000, frac=1.0):
    '''calculates boostrap mean and standard error for connection porbability for presynpatic neurons
    for a specific layer as a function of the difference in preferred orientation
    
    half_dirs: if true directions between 0 and pi else between 0 and pi
    '''

    #Filter to the connections we want
    conn_from_tunedpre = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[pre_layer, "L23"], tuning=["tuned", "tuned"], 
                                                       proofread=proofread, who="both")


    #Avoid self-cnnnections
    conn_from_tunedpre = conn_from_tunedpre[conn_from_tunedpre["pre_id"] != conn_from_tunedpre["post_id"]]

    #For the normalization of the probability, we need the potential links between neurons 
    #given a delta theta. 
    #We need the number of neurons with that delta to compute those 
    tunedlayer = fl.filter_neurons(v1_neurons, tuning="tuned", layer=pre_layer, proofread=proofread[0])
    pre_units_by_angle = tunedlayer["pref_ori"].value_counts().sort_index().values 
    tunedlayer = fl.filter_neurons(v1_neurons, tuning="tuned", layer="L23", proofread=proofread[1])
    post_units_by_angle = tunedlayer["pref_ori"].value_counts().sort_index().values 

    print(pre_units_by_angle)


    #Prepare variables for computing statistics 
    if half:
        p_mean = np.zeros(nangles//2)
        p_std  = np.zeros(nangles//2)

        #To compute the number of neurons with a certain dtheta in the whole system
        #Offset is so that normalization[0] points at the -angle+1 value, so that everything is ordered
        #For 6 angles, the values are -2, -1, 0, 1, 2, 3, so we have to displace +2 = 6/2-1
        normalization = np.zeros(nangles//2)
        offset = nangles//4-1
        limit_angle = nangles//2

        #Bins are done so that Pandas's value_counts gets the integer values. 
        #Ex. counting in the interval [-4.5, -3.5] will give me the amount of times that -3 is present.
        #This method has the advantage of yielding 0 if no -3 is there. Just calling value_counts would omit it, giving problems
        bins = np.arange(-offset-0.5, 0.5 + 1 + nangles//4)
    else:
        #Same, but for directions...
        p_mean = np.zeros(nangles)
        p_std  = np.zeros(nangles)

        normalization = np.zeros(nangles)
        offset = nangles//2-1
        limit_angle = nangles

        bins = np.arange(-offset-0.5, 0.5 + 1 + nangles//2)

    #The number of potential links is just multiplying the number of elements in each group
    for i in range(limit_angle):
        for j in range(limit_angle):
            dtheta = au.signed_angle_dist(i, j, half=half)
            normalization[dtheta+offset] += pre_units_by_angle[i] * post_units_by_angle[j]

    #Bootstrap sampling
    for i in range(n_samps):
        boot_samp = conn_from_tunedpre.sample(frac = frac, replace = True)

        #Calculate how many connections we have per each delta_theta, using the bins as explained before.
        #sort is False so we get them ordered by the index 
        boot_diffs = boot_samp["delta_ori"].value_counts(bins=bins).sort_index().values
        prob = boot_diffs / normalization 

        #Add to the statistics
        p_mean += prob
        p_std  += prob**2

    #Finish averages
    p_mean /= n_samps
    p_std  /= n_samps

    #Return the results as a dataframe normalized by p(dtheta=0)
    return pd.DataFrame({'mean':p_mean , 'std':np.sqrt(p_std - p_mean**2)})

def bootstrap_prob_A2B(v1_neurons, v1_connections, layer=[None, None], tuning=[None,None], cell_type=[None, None], proofread=[None, None], half=True, nangles=16, n_samps=1000):
    '''calculates boostrap mean and standard error for connection porbability for presynpatic neurons
    for a specific layer as a function of the difference in preferred orientation
    
    half_dirs: if true directions between 0 and pi else between 0 and pi
    '''

    #Filter the pre and postsynaptic neurons that we want, and get the connections that corresponds to those
    neurons_pre = fl.filter_neurons(v1_neurons, layer=layer[0], tuning=tuning[0], cell_type=cell_type[0], proofread=proofread[0])
    neurons_post = fl.filter_neurons(v1_neurons, layer=layer[1], tuning=tuning[1], cell_type=cell_type[1], proofread=proofread[1])
    filtered_connections = fl.synapses_by_id(v1_connections, pre_ids=neurons_pre["id"], post_ids=neurons_post["id"], who="both")

    #Take out self-connections
    mask_selfcon = filtered_connections["pre_id"] == filtered_connections["post_id"]
    filtered_connections = filtered_connections[~mask_selfcon]

    if mask_selfcon.any():
        n_potential_links = len(neurons_pre) * (len(neurons_post)-1)
    else:    
        n_potential_links = len(neurons_pre) * len(neurons_post) 


    if n_potential_links == 0:
        print("Warning: 0 links when bootstrapping probability between populations. Probability was set to -1")
        return -1
    else:
        return len(filtered_connections) / n_potential_links

def bootstrap_prob_A2B_2(v1_neurons, v1_connections, layer=[None, None], tuning=[None,None], cell_type=[None, None], proofread=[None, None], half=True, nangles=16, n_samps=1000):
    '''calculates boostrap mean and standard error for connection porbability for presynpatic neurons
    for a specific layer as a function of the difference in preferred orientation
    
    half_dirs: if true directions between 0 and pi else between 0 and pi
    '''

    #---
    p_mean = 0.0
    p_std = 0.0
    #---

    #Filter the pre and postsynaptic neurons that we want, and get the connections that corresponds to those
    neurons_pre = fl.filter_neurons(v1_neurons, layer=layer[0], tuning=tuning[0], cell_type=cell_type[0], proofread=proofread[0])
    neurons_post = fl.filter_neurons(v1_neurons, layer=layer[1], tuning=tuning[1], cell_type=cell_type[1], proofread=proofread[1])
    filtered_connections = fl.synapses_by_id(v1_connections, pre_ids=neurons_pre["id"], post_ids=neurons_post["id"], who="both")


    #---
    for i in range(n_samps):
    #---
        sampled_conns = filtered_connections.sample(frac=1, replace=True) 

        #Take out self-connections
        #mask_selfcon = filtered_connections["pre_id"] == filtered_connections["post_id"]
        #filtered_connections = filtered_connections[~mask_selfcon]
        mask_selfcon = sampled_conns["pre_id"] == sampled_conns["post_id"]
        sampled_conns = sampled_conns[~mask_selfcon]

        if mask_selfcon.any():
            n_potential_links = len(neurons_pre) * (len(neurons_post)-1)
        else:    
            n_potential_links = len(neurons_pre) * len(neurons_post) 


        if n_potential_links == 0:
            #print("Warning: 0 links when bootstrapping probability between populations. Probability was set to -1")
            #return -1
            p_mean -= 1.0
            p_std   -= 1.0
        else:
            #return len(filtered_connections) / n_potential_links
            prob = len(filtered_connections) / n_potential_links
            p_mean += prob 
            p_std  += prob**2 

    #Finish averages
    p_mean /= n_samps
    p_std  /= n_samps

    #Return the results as a dataframe normalized by p(dtheta=0)
    return pd.DataFrame({'mean':p_mean , 'std':np.sqrt(p_std - p_mean**2)})

                                            
def estimate_conn_prob_functmatch(fm_neurons, fm_connections, proof=["minimum", None], nangles=16, half=True):

    if half:
        offset = nangles//4-1
        limit_angle = nangles//2
    else:
        offset = nangles//2-1
        limit_angle = nangles

    columns_ET = [f"ET_{i}" for i in range(limit_angle)]
    columns_XT = [f"XT_{i}" for i in range(limit_angle)]

    #presynaptic (columns) + postsynaptic (rows) names of indices to build a dataframe
    column_names = columns_ET + ["EU", "I"] + columns_XT + ["XU"]
    row_names = columns_ET + ["EU", "I"] 
    ptable = pd.DataFrame(columns=column_names, index=row_names)

    #Now we declare the combinations of freom-to properties that we have
    #For tuning (inhibitory needs to be not masked, so they have a None)
    T2U= ["tuned", "untuned"]
    U2U= ["untuned", "untuned"]
    U2T= ["untuned", "tuned"]

    I2T = [None, "tuned"]
    I2U = [None, "untuned"]
    T2I = ["tuned", None]
    U2I = ["untuned", None]

    #also for cell type
    E2E = ["exc", "exc"]
    E2I = ["exc", "inh"]
    I2I = ["inh", "inh"]
    I2E = ["inh", "exc"]

    #And layer
    L23toL23 = ["L23", "L23"]
    L4toL23 = ["L4", "L23"]

    #Proofread only presynaptic neurons
    #proof = [pre_proofread, None]


    #Let's construct first the values that point to the untuned neurons, i.e., the inh population
    #and the untuned excitatory. For these cases, we do not need to specify the angle of the tune population,
    #that works as a whole. Thus, there's 5 different presynaptic populations 
    #The first two ones will need to be copied for every angle, so store the number in a variable for norw
    prob_ET_2_EU = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=E2E, tuning=T2U, proofread=proof)
    prob_XT_2_EU = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L4toL23, cell_type=E2E, tuning=T2U, proofread=proof)
    #For the other we can assign it directly. Note that notation is ptable[post, pre] 
    ptable.loc["EU", "EU"] = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=E2E, tuning=U2U, proofread=proof)
    ptable.loc["EU", "XU"] = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L4toL23, cell_type=E2E, tuning=U2U, proofread=proof)
    ptable.loc["EU", "I"] = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=I2E, tuning=I2U, proofread=proof)

    #Fill the previously computed value for all angles, as this is angle-independent
    for angle in range(limit_angle):
        ptable.loc["EU", f"ET_{angle}"] = prob_ET_2_EU
        ptable.loc["EU", f"XT_{angle}"] = prob_XT_2_EU


    #Same, but now post is the I population
    prob_ET_2_I = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=E2I, tuning=T2I, proofread=proof)
    prob_XT_2_I = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L4toL23, cell_type=E2I, tuning=T2I, proofread=proof)
    ptable.loc["I", "EU"] = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=E2I, tuning=U2I, proofread=proof)
    ptable.loc["I", "XU"] = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L4toL23, cell_type=E2I, tuning=U2I, proofread=proof)
    ptable.loc["I", "I"] = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=I2I, proofread=proof) #no fuilter for tuning is needed here

    for angle in range(limit_angle):
        ptable.loc["I", f"ET_{angle}"] = prob_ET_2_I
        ptable.loc["I", f"XT_{angle}"] = prob_XT_2_I

    #Finally, to the untuned populations to the tuned E population. Here all of them have to be stored in variables to apply to multiple rows later
    prob_EU_2_ET = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=E2E, tuning=U2T, proofread=proof)
    prob_XU_2_ET = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L4toL23,  cell_type=E2E, tuning=U2T, proofread=proof)
    prob_I_2_ET  = bootstrap_prob_A2B(fm_neurons, fm_connections, layer=L23toL23, cell_type=I2E, tuning=I2T, proofread=proof) #no fuilter for tuning is needed here

    for angle in range(limit_angle):
        ptable.loc[f"ET_{angle}", "EU"] = prob_EU_2_ET
        ptable.loc[f"ET_{angle}", "XU"] = prob_XU_2_ET
        ptable.loc[f"ET_{angle}", "I"] = prob_I_2_ET

    #For the connections between tuned neurons we will need the angle of both. 
    #First, compute the submatrix of probabilities as a function of the angle for each layer.
    prob_T2T_L23 = bootstrap_prob_tuned2tuned(fm_neurons, fm_connections, pre_layer='L23', half=half, proofread=proof)
    prob_T2T_L4  = bootstrap_prob_tuned2tuned(fm_neurons, fm_connections, pre_layer='L4', half=half, proofread=proof)

    #Now, loop over the angles and get the probability that corresponds to a certain dtheta, which is assigned to the table
    for i in range(limit_angle):
        for j in range(limit_angle):
            dtheta = au.signed_angle_dist(i, j, half=half)
            ptable.loc[f"ET_{j}", f"ET_{i}"] = prob_T2T_L23.loc[dtheta+offset, "mean"]
            ptable.loc[f"ET_{j}", f"XT_{i}"] = prob_T2T_L4.loc[dtheta+offset, "mean"]

    #and ready!
    return ptable


def estimate_conn_prob_connectomics(v1_neurons, v1_connections, proof = ["minimum", None]):

    #Similar to function above, but way simpler. Define pre (columns) and post (rows) populations and generate a dataframe
    column_names = ["E", "I", "X"]
    row_names = ["E", "I"]

    ptable = pd.DataFrame(columns=column_names, index=row_names)

    #Handy filters that we will need all the time
    E2E = ["exc", "exc"]
    E2I = ["exc", "inh"]
    I2I = ["inh", "inh"]
    I2E = ["inh", "exc"]

    L23toL23 = ["L23", "L23"]
    L4toL23 = ["L4", "L23"]

    #proof = [pre_proofread, None]

    #Get the numbers between the different populations
    ptable.loc["E", "E"] = bootstrap_prob_A2B(v1_neurons, v1_connections, layer=L23toL23, cell_type=E2E, proofread=proof)
    ptable.loc["E", "I"] = bootstrap_prob_A2B(v1_neurons, v1_connections, layer=L23toL23, cell_type=I2E, proofread=proof)
    ptable.loc["E", "X"] = bootstrap_prob_A2B(v1_neurons, v1_connections, layer=L4toL23, cell_type=E2E, proofread=proof)

    ptable.loc["I", "E"] = bootstrap_prob_A2B(v1_neurons, v1_connections, layer=L23toL23, cell_type=E2I, proofread=proof)
    ptable.loc["I", "I"] = bootstrap_prob_A2B(v1_neurons, v1_connections, layer=L23toL23, cell_type=I2I, proofread=proof)
    ptable.loc["I", "X"] = bootstrap_prob_A2B(v1_neurons, v1_connections, layer=L4toL23, cell_type=E2I, proofread=proof)

    return ptable

def estimate_conn_prob_connectomics_2(v1_neurons, v1_connections, proof = ["minimum", None], n_samps=100):

    #Similar to function above, but way simpler. Define pre (columns) and post (rows) populations and generate a dataframe
    column_names = ["E", "I", "X"]
    row_names = ["E", "I"]

    ptable_mean = pd.DataFrame(0., columns=column_names, index=row_names)
    ptable_std  = pd.DataFrame(0., columns=column_names, index=row_names)

    #Handy filters that we will need all the time
    E2E = ["exc", "exc"]
    E2I = ["exc", "inh"]
    I2I = ["inh", "inh"]
    I2E = ["inh", "exc"]

    L23toL23 = ["L23", "L23"]
    L4toL23 = ["L4", "L23"]

    #proof = [pre_proofread, None]

    for i in range(n_samps):

        sampled_conns = v1_connections.sample(frac=1, replace=True)

        #Get the numbers between the different populations
        p = bootstrap_prob_A2B(v1_neurons, sampled_conns, layer=L23toL23, cell_type=E2E, proofread=proof)
        ptable_mean.loc["E", "E"] += p 
        ptable_std.loc["E", "E"]  += p**2 

        p = bootstrap_prob_A2B(v1_neurons, sampled_conns, layer=L23toL23, cell_type=I2E, proofread=proof)
        ptable_mean.loc["E", "I"] += p 
        ptable_std.loc["E", "I"] += p**2 

        p = bootstrap_prob_A2B(v1_neurons, sampled_conns, layer=L4toL23, cell_type=E2E, proofread=proof)
        ptable_mean.loc["E", "X"] += p
        ptable_std.loc["E", "X"]  += p**2

        p = bootstrap_prob_A2B(v1_neurons, sampled_conns, layer=L23toL23, cell_type=E2I, proofread=proof)
        ptable_mean.loc["I", "E"] += p
        ptable_std.loc["I", "E"]  += p**2

        p = bootstrap_prob_A2B(v1_neurons, sampled_conns, layer=L23toL23, cell_type=I2I, proofread=proof)
        ptable_mean.loc["I", "I"] += p
        ptable_std.loc["I", "I"]  += p**2
        
        p = bootstrap_prob_A2B(v1_neurons, sampled_conns, layer=L4toL23, cell_type=E2I, proofread=proof)
        ptable_mean.loc["I", "X"] += p
        ptable_std.loc["I", "X"]  += p**2

    ptable_mean /= n_samps
    ptable_std  /= n_samps

    #Abs is used because std of case without links is 0, so the difference might be negative due to machine error
    ptable_std = np.sqrt(np.abs(ptable_std - ptable_mean**2))

    return ptable_mean, ptable_std  


def connections_per_population(v1_neurons, v1_connections, nangles=8):
    """
    Constructs a dictionary in which the entry B<A has the connections from population A to  population B. 
    If both A and B are tuned, it furthermore splits connections by difference in pref ori, being stored as
    B<A_deltaX, being X the preferred orientation obtained from au.angle_dist
    """

    connections_by_group = {}

    #Loop over the populations
    for pre in 'EIX':
        layer_pre = 'L4' if pre=='X' else 'L23'
        cell_pre = 'inh' if pre=='I' else 'exc' 
        for post in 'EIX':
            layer_post = 'L4' if post=='X' else 'L23'
            cell_post = 'inh' if post=='I' else 'exc' 

            connections_by_group[f"{post}<{pre}"] = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[layer_pre, layer_post], cell_type=[cell_pre, cell_post], who='both')['syn_volume'] 

    return connections_by_group


def prob_symmetric_links(v1_neurons_, v1_connections_, half=True, nangles=16):
    ''' 
    Computes the relative probability of having symmetric links for each angle

    half_dirs: if true directions between 0 and pi else between 0 and pi
    '''

    #Ensure we are using only the matched neurons. If this does not happen, the
    #adjacency matrix will be tooooo large
    v1_neurons = fl.filter_neurons(v1_neurons_, tuning="matched")
    v1_connections = fl.filter_connections(v1_connections_, tuning="matched")

    #Select connections with presynaptic neurons in the selected layer 
    tunedlayer = fl.filter_neurons(v1_neurons, tuning="tuned", layer="L23")
    conn_from_tunedpre = fl.filter_connections(v1_neurons, v1_connections, tuning="tuned", who="both")
    conn_from_tunedpre = fl.filter_connections(v1_neurons, conn_from_tunedpre, layer="L23", who="pre")

    #Get the double links from the adjacency matrix
    tuned_ids = tunedlayer["id"].values
    m = loader.get_adjacency_matrix(v1_neurons, v1_connections)
    m = m[np.ix_(tuned_ids, tuned_ids)]

    rows, cols = np.where(m * m.transpose() > 0)

    #Now filter the connections table to include only the double links
    new_rows = []
    for i in range(len(cols)):
        #MWhen we filtered indices changed. Recover the real ones
        id1 = tuned_ids[rows[i]]
        id2 = tuned_ids[cols[i]]

        #Make sure our indices are either pre or post
        is_pre = (conn_from_tunedpre["pre_id"] == id1) | (conn_from_tunedpre["pre_id"] == id2)
        is_post = (conn_from_tunedpre["post_id"] == id1) | (conn_from_tunedpre["post_id"] == id2)
        for row in conn_from_tunedpre[is_pre & is_post].values:
            new_rows.append(row) 

    #Finish and make all differences positive (symmetric links, we do not care about angle)
    new_rows = np.array(new_rows)
    symmetric_only = pd.DataFrame(new_rows, columns=conn_from_tunedpre.columns)
    symmetric_only = symmetric_only.iloc[::2]
    symmetric_only["delta_ori"] = symmetric_only["delta_ori"].abs()

    #Prepare variables for computing statistics 
    if half:
        #To compute the number of neurons with a certain dtheta in the whole system
        #Offset is so that normalization[0] points at the -angle+1 value, so that everything is ordered
        #For 6 angles, the values are -2, -1, 0, 1, 2, 3, so we have to displace +2 = 6/2-1
        limit_angle = nangles//2
        normalization = np.zeros(limit_angle//2 + 1)

        #Bins are done so that Pandas's value_counts gets the integer values. 
        #Ex. counting in the interval [-4.5, -3.5] will give me the amount of times that -3 is present.
        #This method has the advantage of yielding 0 if no -3 is there. Just calling value_counts would omit it, giving problems
        bins = np.arange(-0.5, 0.5 + limit_angle//2+1)
    else:
        limit_angle = nangles
        normalization = np.zeros(limit_angle//2 + 1)

        bins = np.arange(-0.5, 0.5 + limit_angle//2+1)

    #For the normalization of the probability, we need the potential links between neurons 
    #given a delta theta 
    ids_symmetric = symmetric_only["pre_id"].unique()
    n_units_by_angle = tunedlayer[tunedlayer["id"].isin(ids_symmetric)]["pref_ori"].value_counts()

    #For all possible neuron's angles, get the delta theta and compute the amount
    #of links for that delta theta. 
    for i in range(limit_angle):

        #for delta=0, we exhaust one link per each node we visit, 
        #giving always (n-1) + (n+2) + ... + 2 + 1 potential links 
        normalization[0] += (n_units_by_angle[i]-1) * n_units_by_angle[i] // 2

        #If angles are not equal, the number of potential links is just multiplying the 
        #number of elements in each group
        for j in range(limit_angle):
            if i != j:
                dtheta = au.angle_dist(i, j)
                normalization[dtheta] += n_units_by_angle[i] * n_units_by_angle[j]

    #Last distance is computed twice
    normalization[-1] = normalization[-1] // 2 

    #Calculate how many connections we have per each delta_theta, using the bins as explained before.
    #sort is False so we get them ordered by the index 
    n_double_links = symmetric_only["delta_ori"].value_counts(bins=bins)
    n_double_links = n_double_links.reindex(np.arange(limit_angle//2+1))
    prob = n_double_links.sort_index() / normalization

    #Add the others for symmetry!
    for angle in range(-limit_angle//2, 0):
        prob[angle] = prob[-angle]

    return prob.sort_index()

def get_fraction_populations(units):

    fractions = {}
    ne = len(fl.filter_neurons(units, cell_type='exc', layer='L23', tuning='unmatched'))
    ni = len(fl.filter_neurons(units, cell_type='inh', layer='L23', tuning='unmatched'))
    nx = len(fl.filter_neurons(units, cell_type='exc', layer='L4',  tuning='unmatched'))

    nt = ne+ni+nx

    fractions["E"] = ne/nt
    fractions["I"] = ni/nt
    fractions["X"] = nx/nt

    nem = len(fl.filter_neurons(units, cell_type='exc', layer='L23', tuning='matched'))
    nxm = len(fl.filter_neurons(units, cell_type='exc', layer='L4',  tuning='matched'))

    tuned_e = fl.filter_neurons(units, cell_type='exc', layer='L23', tuning='tuned')
    tuned_x = fl.filter_neurons(units, cell_type='exc', layer='L4',  tuning='tuned')
    net = len(tuned_e)
    nxt = len(tuned_x)

    fractions["ET"] = fractions['E'] * net/nem
    fractions["EU"] = fractions['E'] * (nem-net)/nem

    fractions["XT"] = fractions['X'] * nxt/nxm 
    fractions["XU"] = fractions['X'] * (nxm-nxt)/nxm 

    n_angle_e = tuned_e['pref_ori'].value_counts()
    n_angle_x = tuned_x['pref_ori'].value_counts()

    for i in range(8):
        fractions[f'ET_{i}'] = fractions['ET'] * n_angle_e[i]/net
        fractions[f'XT_{i}'] = fractions['XT'] * n_angle_x[i]/nxt 

    return pd.Series(fractions)

def prob_conn_diffori(v1_neurons, v1_connections, half=True, proofread=['ax_clean', None], frac=1.0, n_samps=1000):
    """
    Computes the connection probability between neurons depending on the difference of orientation between them.

    Parameters: requires the dataframe of neurons's properties, as well as the synapses' properties. 
    """

    #Extract bootstrapped stats
    l23_boots = bootstrap_prob_tuned2tuned(v1_neurons, v1_connections, pre_layer='L23', half=half, proofread=proofread, frac=frac, n_samps=n_samps)
    l4_boots = bootstrap_prob_tuned2tuned(v1_neurons, v1_connections, pre_layer='L4', half=half, proofread=proofread, frac=frac, n_samps=n_samps)

    return l23_boots, l4_boots

def cumul_dist(data, n_bins):

    histv, bins = np.histogram(data, bins = n_bins)
    cumulhist = np.cumsum(histv)

    return cumulhist, bins[1:]

def cumulative_probconn(v1_neurons, v1_connections, angles_list):
    """
    Cumulative distribution of the connectivity between neurons with perpendicular
    orientation, from layers 2/3 or 4, to L2/3
    """

    synapse_tuning = utl.tuning_encoder(v1_neurons, v1_connections)
    tuned_connections = v1_connections[synapse_tuning['pre_tuned'] & synapse_tuning["post_tuned"]]
    
    normcstr = np.mean(v1_connections[v1_connections['pre_layer'] == 'L23']['syn_volume'])
    
    sub1l4 =tuned_connections[(tuned_connections['delta_ori'] == angles_list[0])& 
                          (tuned_connections['pre_layer'] == 'L4')]['syn_volume'].values/normcstr 
    sub2l4 =tuned_connections[(tuned_connections['delta_ori'] == angles_list[1])& 
                          (tuned_connections['pre_layer'] == 'L4')]['syn_volume'].values/normcstr 
    sub1l23 =tuned_connections[(tuned_connections['delta_ori'] == angles_list[2])& 
                           (tuned_connections['pre_layer'] == 'L23')]['syn_volume'].values/normcstr 
    sub2l23 =tuned_connections[(tuned_connections['delta_ori'] == angles_list[3])& 
                           (tuned_connections['pre_layer'] == 'L23')]['syn_volume'].values/normcstr 
    
    logbins = np.logspace(-2.3, 1.5, 100) 

    ch1, b1 = cumul_dist(sub1l4, logbins)

    ch2, b2 = cumul_dist(sub2l4, logbins)

    ch3, b3 = cumul_dist(sub1l23, logbins)

    ch4, b4 = cumul_dist(sub2l23, logbins)

    return [[ch1, b1], [ch2, b2], [ch3, b3], [ch4, b4]]