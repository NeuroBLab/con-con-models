import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ccmodels.analysis.simulators import bootstrap_conn_prob, tpo_po_simulator_new,bootstrap_layerinput_proportions
from ccmodels.analysis.utils import tuning_segmenter, compute_cur_delta_dist, get_current_normalization
from ccmodels.analysis.aggregators import flatten_orientation_current, all_inputs_aggregator2, cumul_dist




def figure_saver(fig,name,width_cm, height_cm, save_path):
    '''Allows to save an image in pdf format specifing size in cm'''

    w_inch = width_cm/2.54
    h_inch = height_cm/2.54

    fig.set_size_inches(w_inch, h_inch) 
    fig.savefig(f'{save_path}{name}.pdf', format='pdf', bbox_inches='tight')

# --- Auxiliary functions for Figure 1 --- 

def get_number_connections(data_location = '../con-con-models/data', layer234_only=True):
    '''Reads in appropriate data and prepares it for plotting by calculating counts'''

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


def get_propotion_connections(data_location = '../con-con-models/data'):
    '''
    Performs a bootstrap analysis on the proportion of inputs from L2/3 and L4 neurons that are received by
    L2/3 neurons that are proofread VS that are not proofread
    '''

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


# --- Auxiliary functions for Figure 2 --- 


def prob_conectivity_tuned_untuned(v1_connections, nsamples=100):
    """
    Perform a number nsamples of bootstrap samples of the connection probability, and returns a matrix
    of probability of connection between tuned/untuned populations of layers 2/3 and 4 to layer 2/3.
    """
    #Connection probability between combinations of tuned and untuned inputs/outputs
    #Rows: L2/3T, L2/3U (post synaptic)
    #Columns: L2/3T, L2/3U,  L4T, L4U (pre synaptic)
    sampled_probabilities = np.empty((2, 4))

    for i in tqdm(range(nsamples)):
        #Sample a number of connections with repetitions
        v1_connections_samp = v1_connections.sample(v1_connections.shape[0], replace = True)

        #Separate the samples in both tuned/untuned and layer.
        l4_combinations, l23_combinations = tuning_segmenter(v1_connections_samp)

        #L4 -> L2/3
        #Normalising constants, total number of connections
        layer_4_tuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 1)
                                & (v1_connections_samp['pre_layer'] == 'L4')].shape[0]

        layer_4_untuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 0)
                                & (v1_connections_samp['pre_layer'] == 'L4')].shape[0]

        #Compute the averages!
        #L4T -> L2/3T
        sampled_probabilities[0,2] += (l4_combinations[0].shape[0]/layer_4_tuned_out)
        #L4T -> L2/3U
        sampled_probabilities[1,2] += (l4_combinations[1].shape[0]/layer_4_untuned_out)
        #L4U -> L2/3T
        sampled_probabilities[0,3] += (l4_combinations[3].shape[0]/layer_4_tuned_out)
        #L4U -> L2/3U
        sampled_probabilities[1,3] += (l4_combinations[2].shape[0]/layer_4_untuned_out)


        #L2/3 -> L2/3
        #Normalising constants
        layer_23_tuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 1)
                                & (v1_connections_samp['pre_layer'] == 'L2/3')].shape[0]

        layer_23_untuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 0)
                                & (v1_connections_samp['pre_layer'] == 'L2/3')].shape[0]

        #Compute the averages!
        #L2/3T -> L2/3T
        sampled_probabilities[0,0] += (l23_combinations[0].shape[0]/layer_23_tuned_out)
        #L2/3T -> L2/3U
        sampled_probabilities[1,0] += (l23_combinations[1].shape[0]/layer_23_untuned_out)
        #L2/3U -> L2/3T
        sampled_probabilities[0,1] += (l23_combinations[3].shape[0]/layer_23_tuned_out)
        #L2/3U -> L2/3U
        sampled_probabilities[1,1] += (l23_combinations[2].shape[0]/layer_23_untuned_out)

    #Finish average and return
    return sampled_probabilities/nsamples

def strength_tuned_untuned(v1_connections):
    """
    Returns the matrix of connection strength from layer 2/3 and 4 to layer 2/3,
    normalized to the strenght of recurrent layer 2/3 connections.
    """

    #Rows: L2/3T, L2/3U (post synaptic)
    #Columns: L2/3T, L2/3U,  L4T, L4U (pre synaptic)
    sampled_strenghts = np.empty((2, 4))

    #Extract tuning combinations
    l4_combinations, l23_combinations = tuning_segmenter(v1_connections)

    #Normalization
    normcstr = np.mean(v1_connections[v1_connections['pre_layer'] == 'L2/3']['size'])

    #Fill our matrix accordingly. See also the previous function
    sampled_strenghts[0,0] = np.mean(l23_combinations[0]['size']) #From L2/3T to L2/3 T,U
    sampled_strenghts[1,0] = np.mean(l23_combinations[1]['size'])
    sampled_strenghts[0,1] = np.mean(l23_combinations[2]['size']) #From L2/3U to L2/3 T,U
    sampled_strenghts[1,1] = np.mean(l23_combinations[3]['size'])

    sampled_strenghts[0,2] = np.mean(l4_combinations[0]['size']) #From L4 T,U to L2/3
    sampled_strenghts[1,2] = np.mean(l4_combinations[1]['size'])
    sampled_strenghts[0,3] = np.mean(l4_combinations[2]['size'])
    sampled_strenghts[1,3] = np.mean(l4_combinations[3]['size'])

    return sampled_strenghts/normcstr


def prob_conn_diffori(v1_connections):
    """
    Computes the connection probability between neurons depending on the difference of orientation between them.
    """

    #Identify tuned neurons
    tuned_neurons = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                   (v1_connections['pre_type']!= 'not_selective')]

    #Extract bootstrapped stats
    l23_boots = bootstrap_conn_prob(tuned_neurons,pre_layer='L2/3')
    l4_boots = bootstrap_conn_prob(tuned_neurons, pre_layer='L4')

    return l23_boots, l4_boots


def cumulative_probconn(v1_connections, angles_list):
    """
    Cumulative distribution of the connectivity between neurons with perpendicular
    orientation, from layers 2/3 or 4, to L2/3
    """

    #examp_vals = [0, 1.570796, 0, 1.570796]
    tuned_neurons = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                   (v1_connections['pre_type']!= 'not_selective')]
    
    normcstr = np.mean(v1_connections[v1_connections['pre_layer'] == 'L2/3']['size'])
    
    sub1l4 =tuned_neurons[(tuned_neurons['delta_ori_constrained'] == angles_list[0])& 
                          (tuned_neurons['pre_layer'] == 'L4')]['size'].values/normcstr 
    sub2l4 =tuned_neurons[(tuned_neurons['delta_ori_constrained'] == angles_list[1])& 
                          (tuned_neurons['pre_layer'] == 'L4')]['size'].values/normcstr 
    sub1l23 =tuned_neurons[(tuned_neurons['delta_ori_constrained'] == angles_list[2])& 
                           (tuned_neurons['pre_layer'] == 'L2/3')]['size'].values/normcstr 
    sub2l23 =tuned_neurons[(tuned_neurons['delta_ori_constrained'] == angles_list[3])& 
                           (tuned_neurons['pre_layer'] == 'L2/3')]['size'].values/normcstr 
    
    logbins = np.logspace(-2.3, 1.5, 100) 

    ch1, b1 = cumul_dist(sub1l4, logbins)

    ch2, b2 = cumul_dist(sub2l4, logbins)

    ch3, b3 = cumul_dist(sub1l23, logbins)

    ch4, b4 = cumul_dist(sub2l23, logbins)

    return [[ch1, b1], [ch2, b2], [ch3, b3], [ch4, b4]]

# --- Auxiliary functions for figure 3 ---- 

"""
def get_input_frompresynaptic(connections_to_post, dir_range="full", input_name="shifted_current"):
    flattened = flatten_orientation_current(connections_to_post, "new_dirs", input_name, dir_range)
    flattened = flattened.rename(columns={"dirs" : "new_dirs", "norm_cur" : input_name})

    #Input to the neuron, for each angle. We select only the current and sum over, so result is pd.Series with index new_dirs
    return flattened.groupby("new_dirs")[input_name].sum().reset_index() 
"""

def get_input_frompresynaptic(connections_to_post, dir_range="full", input_name="shifted_current"):
    current_to_post = np.zeros(16)
    for current in connections_to_post["shifted_current"]:
        current += np.array(current_to_post)
    
    return pd.DataFrame({input_name : current}) 



def compute_avg_inpt_current(v1_connections, proofread_input_n, dir_range):
    """
    Computes the average input current to neurons in the L2/3, as well as the proportion of it
    arriving from recurrent interactions and L4
    """

    #Use both tuned and unted neurons, and shuffle untuned ones
    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective')]
    mask_untuned = tuned_outputs["pre_type"] == "not_selective"
    tuned_outputs.loc[mask_untuned, "shifted_current"] = tuned_outputs.loc[mask_untuned, "shifted_current"].apply(lambda x : np.random.choice(x, len(x), replace=False)) 
    

    #Get presynaptic neurons depending on layer
    l23_t = tuned_outputs[tuned_outputs['pre_layer'] == 'L2/3']
    l23_t = l23_t.copy()
    l4_t = tuned_outputs[tuned_outputs['pre_layer'] == 'L4']
    l4_t = l4_t.copy()
    
    #Calculate aggregated currents
    avg_input_l23, _, _= all_inputs_aggregator2(l23_t, 'shifted_current', 'new_dirs', grouping = 'mean',  dir_range = dir_range)
    avg_input_l4, _, _= all_inputs_aggregator2(l4_t, 'shifted_current', 'new_dirs', grouping = 'mean',   dir_range = dir_range)

    # calculate average number of L2/3 inputs and number of L4 inputs
    avg_connections_l23 = np.mean(proofread_input_n[proofread_input_n['cortex_layer'] == 'L2/3']['n_connections'])
    avg_connections_l4 = np.mean(proofread_input_n[proofread_input_n['cortex_layer'] == 'L4']['n_connections'])
    tot = avg_connections_l23+avg_connections_l4

    #Proportions of inputs from each layer
    propl23 = avg_connections_l23/tot
    propl4 = avg_connections_l4/tot

    return avg_input_l23, propl23, avg_input_l4, propl4




def single_synapse_current(v1_connections, n_neurons, seed=4, dir_range="full", also_L4=True):
    """
    Compute real input from the data using just a single neuron. We use n_neurons both from layer 2/3 and 4 
    """

    #Use only tuned neurons
    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & (v1_connections['pre_type']!= 'not_selective')]
    
    #Get which neurons are in each layer
    tuned_l23 = tuned_outputs[tuned_outputs["pre_layer"] == "L2/3"]

    #Get some neurons from each one of the layers
    np.random.seed(seed)
    if type(n_neurons) == int:
        ids_l23 = tuned_l23.sample(n_neurons, replace=False)
    else:
        ids_l23 = tuned_l23.iloc[n_neurons,:]

    maxcurr = get_current_normalization(tuned_outputs["shifted_current"])

    ids_l23["shifted_current"] /= maxcurr 

    #Repeat for L4, if we want neurons from there too
    #and then return
    if also_L4:
        tuned_l4  = tuned_outputs[tuned_outputs["pre_layer"] == "L4"]
        ids_l4  = tuned_l4.sample(n_neurons, replace=False)
        ids_l4["shifted_current"]  /= maxcurr 

        return {"L2/3" : ids_l23[["new_dirs", "shifted_current"]], 
                "L4" : ids_l4[["new_dirs", "shifted_current"]]}
    else:
        #Return the just L2/3
        return {"L2/3" : ids_l23[["new_dirs", "shifted_current"]]} 



def compute_distrib_diffrate_allsynapses(v1_connections, dir_range="full"):

    if dir_range=="full":
        index_zero = 7 
        index_pihalf = 11
    else:
        index_zero = 0 
        index_pihalf = 4 

    #Use only tuned neurons
    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & (v1_connections['pre_type']!= 'not_selective')]
    
    #Get which neurons are in each layer
    tuned_l23 = tuned_outputs[tuned_outputs["pre_layer"] == "L2/3"]
    tuned_l4  = tuned_outputs[tuned_outputs["pre_layer"] == "L4"]

    #Get number of angles, and set up a Dataframe based on it, that we will fill
    #Keep each result in a separate table depending on layer of presynaptic
    diffs = {"L2/3" : [],
              "L4"   : []} 
    checked = False

    maxcurr = get_current_normalization(tuned_outputs["shifted_current"])
    #Compute the inputs for all the selected neurons. Do it separately for each layer
    for layer, allsynapses in zip(["L2/3", "L4"], [tuned_l23, tuned_l4]):
        for current in allsynapses["shifted_current"]:

            #Normalize currents
            norm_curr = current/maxcurr 

            #Difference between pi/half and 0
            diffs[layer].append(norm_curr[index_zero] - norm_curr[index_pihalf]) 

    return diffs 



def compute_inpt_bootstrap2(v1_connections, nsamples=250, dir_range="full", seed=4):
    """
    Bootstrap input per each neuron in the dataset, using its own connectivity 
    """

    #Use only tuned neurons
    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & (v1_connections['pre_type']!= 'not_selective')]

    mask_untuned = tuned_outputs["pre_type"] == "not_selective"
    tuned_outputs[mask_untuned]["shifted_current"] = tuned_outputs[mask_untuned]["shifted_current"].apply(np.random.shuffle)
    tuned_outputs[mask_untuned]["shifted_current2"] = tuned_outputs[mask_untuned]["shifted_current2"].apply(np.random.shuffle)
    
    #Select ids of the post synaptic neurons to study
    np.random.seed(seed)
    neuron_ids = tuned_outputs["post_id"].values
    angles = tuned_outputs["new_dirs"].values[0]

    prob_pref_ori = np.zeros(len(angles)) 
    for id in neuron_ids:
        #Grab all rows of this post synaptic neuron 
        presynpatics = tuned_outputs[tuned_outputs["post_id"] == id]

        #Bootstrap the connections and get the input to the neuron 
        preboostrap = presynpatics.sample(nsamples, replace=True)
        total_activity = get_input_frompresynaptic(preboostrap, dir_range=dir_range)

        rate = total_activity["shifted_current"].values

        idx_prefrd_ori = np.argmax(rate)
        prob_pref_ori[idx_prefrd_ori] += 1

    prob_pref_ori /= np.sum(prob_pref_ori)
    return np.array(angles), prob_pref_ori 



def compute_inpt_bootstrap(tuned_connections, nexperiments, nsamples=250, dir_range="full", seed=4, reshuffle_all=False):

    #Check if we do the actual thing, or if we try with a reshuffled model for comparison.
    if not reshuffle_all:
        mask_untuned = tuned_connections["pre_type"] == "not_selective"
        tuned_connections.loc[mask_untuned, "shifted_current"] = tuned_connections.loc[mask_untuned, "shifted_current"].apply(lambda x : np.random.choice(x, len(x), replace=False)) 
    else:
        tuned_connections.loc[:, "shifted_current"] = tuned_connections["shifted_current"].apply(lambda x : np.random.choice(x, len(x), replace=False)) 
    
    #Get the angles
    angles = tuned_connections["new_dirs"].values[0]
    #angles = np.arange(0, 2*np.pi, np.pi/8)

    #Prepare to do experiments...
    prob_pref_ori = np.zeros(len(angles)) 
    for i in range(nexperiments): 

        #Sample a bunch of neurons
        neuron_sample = tuned_connections.sample(nsamples, replace=True) 

        #COmpute the current they get 
        current = get_input_frompresynaptic(neuron_sample, dir_range=dir_range, input_name="shifted_current")

        #Use the current to determine the preferred orientation
        idx_prefrd_ori = np.argmax(current["shifted_current"])
        prob_pref_ori[idx_prefrd_ori] += 1
    
    #Normalize
    prob_pref_ori /= np.sum(prob_pref_ori)

    #Return (transform to numpy array)
    return np.array(angles), prob_pref_ori 

def compute_inpt_reshuffled(tuned_connections, nexperiments, nsamples=250, dir_range="full", seed=4):

    tuned_connections["shifted_current"] = tuned_connections["shifted_current"].apply(np.random.shuffle)
    
    #Get the angles
    angles = tuned_connections["new_dirs"].values[0]
    #angles = np.arange(0, 2*np.pi, np.pi/8)

    #Prepare to do experiments...
    prob_pref_ori = np.zeros(len(angles)) 
    for i in range(nexperiments): 

        #Sample a bunch of neurons
        neuron_sample = tuned_connections.sample(nsamples, replace=True) 

        #COmpute the current they get 
        current = get_input_frompresynaptic(neuron_sample, dir_range=dir_range, input_name="shifted_current")

        #Use the current to determine the preferred orientation
        idx_prefrd_ori = np.argmax(current["shifted_current"])
        prob_pref_ori[idx_prefrd_ori] += 1
    
    #Normalize
    prob_pref_ori /= np.sum(prob_pref_ori)

    #Return (transform to numpy array)
    return np.array(angles), prob_pref_ori 



# -------------------------------------------------------------------

























def prepare_c3(v1_connections):

    l23_rows = v1_connections[(v1_connections['pre_type']!= 'not_selective') & (v1_connections['pre_layer'] == 'L2/3')]
    l4_rows = v1_connections[(v1_connections['pre_type']!= 'not_selective') & (v1_connections['pre_layer'] == 'L4')]

    l23_curdelta = compute_cur_delta_dist(l23_rows)
    l4_curdelta = compute_cur_delta_dist(l4_rows)


    l23_curdelta = np.array(l23_curdelta)
    l4_curdelta = np.array(l4_curdelta)

    l4_curdelta =l4_curdelta/np.mean(l23_curdelta)
    l23_curdelta=l23_curdelta/np.mean(l23_curdelta)
    ltot_curdelta=np.concatenate((l4_curdelta,l23_curdelta))

    return ltot_curdelta


def prepare_d3(v1_connections):

    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                   (v1_connections['pre_type']!= 'not_selective')]
    
    untuned_inputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                    (v1_connections['pre_type']== 'not_selective')]


    #Extract a subset of the columns in the activities dataframe
    l4_pre = tuned_outputs[tuned_outputs['pre_layer'] == 'L4'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]
    l23_pre = tuned_outputs[tuned_outputs['pre_layer'] == 'L2/3'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]


    #Extract the subset of untuned inputs
    l4_pre_untuned = untuned_inputs[untuned_inputs['pre_layer'] == 'L4'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]
    l23_pre_untuned = untuned_inputs[untuned_inputs['pre_layer'] == 'L2/3'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]

    n_post = 5000
    n_pre = 250
    n23_pre = int(n_pre*0.8)
    n4_pre = int(n_pre*0.2)

    weighted_simulation = tpo_po_simulator_new(n_post, n23_pre, n4_pre, l4_pre, l23_pre, 
                                            l4_pre_untuned, l23_pre_untuned, weighted = True, pre_profile='both')


    #Generate bins for histogram between 0-np.pi
    dirssamp_abs = np.array(sorted(list(set(np.abs(l4_pre['delta_ori_constrained'].round(3))))))

    bin_size_abs = np.diff(np.array(sorted(list(set(dirssamp_abs)))))

    startabs = dirssamp_abs- (bin_size_abs[0]/2)
    endabs = list(dirssamp_abs+ (bin_size_abs[0]/2))

    endabs = sorted(endabs+[startabs[0]])


    valsabs, binsabs = np.histogram(np.abs(weighted_simulation), density = True, bins = endabs)

    return valsabs, binsabs

def prepare_e3(v1_connections):

    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                   (v1_connections['pre_type']!= 'not_selective')]
    
    untuned_inputs = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                    (v1_connections['pre_type']== 'not_selective')]

    #Extract a subset of the columns in the activities dataframe
    l4_pre = tuned_outputs[tuned_outputs['pre_layer'] == 'L4'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]
    #l4_pre['pre_po'] = l4_pre['pre_po']
    l23_pre = tuned_outputs[tuned_outputs['pre_layer'] == 'L2/3'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]

    #Extract the subset of untuned inputs
    l4_pre_untuned = untuned_inputs[untuned_inputs['pre_layer'] == 'L4'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]
    l23_pre_untuned = untuned_inputs[untuned_inputs['pre_layer'] == 'L2/3'][['pre_po', 'pre_activity', 'pre_orientations', 'delta_ori', 'size', 'shifted_activity', 'new_dirs', 'delta_ori_constrained', 'shifted_current']]

    n_post = 500
    n_pre = 250
    n23_pre = int(n_pre*0.8)
    n4_pre = int(n_pre*0.2)

    mses = []
    ks = []
    for i in tqdm(range(5, 1500, 100)): #1500 is the upper input bound calclated in the dedicated notebook
        n_pre= i

        weighted_simulation = tpo_po_simulator_new(n_post, n23_pre, n4_pre, l4_pre, l23_pre, 
                                            l4_pre_untuned, l23_pre_untuned, weighted = True, pre_profile='both')
        
        mse= np.sum(np.array(weighted_simulation)**2)/len(weighted_simulation)
        mses.append(mse)

        ks.append(i)


    return ks, mses