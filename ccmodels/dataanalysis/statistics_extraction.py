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


def bootstrap_conn_prob(v1_neurons, v1_connections, pre_layer, half=True, nangles=16, n_samps=1000):
    '''calculates boostrap mean and standard error for connection porbability for presynpatic neurons
    for a specific layer as a function of the difference in preferred orientation
    
    half_dirs: if true directions between 0 and pi else between 0 and pi
    '''

    #Select connections with presynaptic neurons in the selected layer 
    #If layer is 2/3 we want to avoid L4, because all neurons there are presyanptic and 
    #therefore they cannot count for the potential connections (for normalization)
    #If it's layer 4, then all neurons can form potential connections, so we use full system.
    if pre_layer == "L23":
        tunedlayer = fl.filter_neurons(v1_neurons, tuning="tuned", layer="L23")
    else:
        tunedlayer = fl.filter_neurons(v1_neurons, tuning="tuned")

    #Filter to the connections we want
    conn_from_tunedpre = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[pre_layer, "L23"], tuning=["tuned", "tuned"], 
                                                       proofread=["decent", None])

    #conn_from_tunedpre = fl.filter_connections_prepost(v1_neurons, v1_connections, layer=[pre_layer, "L23"], tuning=["tuned", "tuned"]) 
    conn_from_tunedpre = conn_from_tunedpre[conn_from_tunedpre["pre_id"] != conn_from_tunedpre["post_id"]]

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

    #For the normalization of the probability, we need the potential links between neurons 
    #given a delta theta 
    n_units_by_angle = tunedlayer["pref_ori"].value_counts().sort_index().values 

    #For all possible neuron's angles, get the delta theta and compute the amount
    #of links for that delta theta. 
    for i in range(limit_angle):
        for j in range(limit_angle):
            #If they are not equal, the number of potential links is just multiplying the 
            #number of elements in each group
            if i != j:
                dtheta = au.signed_angle_dist(i, j, half=half)
                normalization[dtheta+offset] += n_units_by_angle[i] * n_units_by_angle[j]

            else:
                #for delta=0, however, we exhaust one link per each node we visit, 
                #giving always (n-1) + (n+2) + ... + 2 + 1 potential links. Links can 
                #be back and forward, so multiply by 2.
                normalization[offset] += (n_units_by_angle[i]-1) * n_units_by_angle[i] 



    #Bootstrap sampling
    for i in range(n_samps):
        boot_samp = conn_from_tunedpre.sample(frac = 1, replace = True)

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

    #Return the results as a dataframe
    return pd.DataFrame({'mean':p_mean/np.max(p_mean) , 'std':np.sqrt(p_std - p_mean**2)/np.max(p_mean)})

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


def prob_conn_diffori(v1_neurons, v1_connections, half=True):
    """
    Computes the connection probability between neurons depending on the difference of orientation between them.

    Parameters: requires the dataframe of neurons's properties, as well as the synapses' properties. 
    """

    #Extract bootstrapped stats
    l23_boots = bootstrap_conn_prob(v1_neurons, v1_connections, pre_layer='L23', half=half)
    l4_boots = bootstrap_conn_prob(v1_neurons, v1_connections, pre_layer='L4', half=half)

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