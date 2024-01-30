import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ccmodels.analysis.simulators import bootstrap_conn_prob, tpo_po_simulator_new,bootstrap_layerinput_proportions
from ccmodels.analysis.utils import tuning_segmenter, compute_cur_delta_dist
from ccmodels.analysis.aggregators import all_inputs_aggregator2, cumul_dist




def figure_saver(fig,name,width_cm, height_cm, save_path):
    '''Allows to save an image in pdf format specifing size in cm'''

    w_inch = width_cm/2.54
    h_inch = height_cm/2.54

    fig.set_size_inches(w_inch, h_inch) 
    fig.savefig(f'{save_path}{name}.pdf', format='pdf', bbox_inches='tight')


def prepare_c1(data_location = '../con-con-models/data'):
    '''Reads in appropriate data and prepares it for plotting by calculating counts'''

    nonproof_inputs_sample = pd.read_csv(f'{data_location}/nonproof_inputs_sample.csv')
    proof_inputs_sample = pd.read_csv(f'{data_location}/proofread_inputs_sample.csv')

    nonproof_outputs_sample = pd.read_csv(f'{data_location}/nonproof_outputs_sample.csv')
    proof_outputs_sample = pd.read_csv(f'{data_location}/proofread_outputs_sample.csv')


    #Count number of input neurons for each post synaptic neuron
    nonproof_inputs_counts = nonproof_inputs_sample.groupby('post_pt_root_id').count().reset_index()
    proof_inputs_counts = proof_inputs_sample.groupby('post_pt_root_id').count().reset_index()

    #Count number of output neurons for each pre synaptic neuron
    nonproof_outputs_counts = nonproof_outputs_sample.groupby('pre_pt_root_id').count().reset_index()
    proof_outputs_counts = proof_outputs_sample.groupby('pre_pt_root_id').count().reset_index()

    return nonproof_inputs_counts, proof_inputs_counts, nonproof_outputs_counts, proof_outputs_counts


def prepare_d1(data_location = '../con-con-models/data'):
    '''
    Performs a bootstrap analysis on the proportion of inputs from L2/3 and L4 neurons that are received by
    L2/3 neurons that are proofread VS that are not proofread
    '''
    proof_inputs_l23sample = pd.read_csv(f'{data_location}/proof_inputs_l23sample.csv')
    nonproof_inputs_l23sample = pd.read_csv(f'{data_location}/nonproof_inputs_l23sample.csv')


    #Group inputs by postsynaptic neurons and by layer
    layer_groups_nop = proof_inputs_l23sample.groupby(['post_pt_root_id','cortex_layer']).count()
    layer_groups_p = nonproof_inputs_l23sample.groupby(['post_pt_root_id','cortex_layer']).count()

    #Clean the dataframes
    #Proofread
    layer_groups_p = layer_groups_p.reset_index().loc[:, ['post_pt_root_id', 'cortex_layer', 'id']].rename(columns = {'id':'n_connections'})
    #NON-proofread
    layer_groups_nop = layer_groups_nop.reset_index().loc[:, ['post_pt_root_id', 'cortex_layer', 'id']].rename(columns = {'id':'n_connections'})

    boots_propl_proof = bootstrap_layerinput_proportions(layer_groups_p, 'cortex_layer', 'n_connections', 
                                                         layer_labels = ['L2/3', 'L4'], n_iters = 100)
    
    boots_propl_noproof = bootstrap_layerinput_proportions(layer_groups_nop, 'cortex_layer', 'n_connections', 
                                                           layer_labels = ['L2/3', 'L4'], n_iters = 100)
    
    
    return boots_propl_proof, boots_propl_noproof


def prepare_b2(v1_connections):
    #Connection probability between combinations of tuned and untuned inputs/outputs
    l4t_l23t = []
    l4t_l23u = []
    l4u_l23u = []
    l4u_l23t = []

    l23t_l23t = []
    l23t_l23u = []
    l23u_l23u = []
    l23u_l23t = []

    for i in tqdm(range(100)):
        v1_connections_samp = v1_connections.sample(v1_connections.shape[0], replace = True)
        l4_combinations, l23_combinations = tuning_segmenter(v1_connections_samp)

        #L4 -> L2/3
        #Normalising constants
        layer_4_tuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 1)
                                & (v1_connections_samp['pre_layer'] == 'L4')].shape[0]

        layer_4_untuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 0)
                                & (v1_connections_samp['pre_layer'] == 'L4')].shape[0]

        #Probabilities
        l4t_l23t.append(l4_combinations[0].shape[0]/layer_4_tuned_out)
        l4t_l23u.append(l4_combinations[1].shape[0]/layer_4_untuned_out)
        l4u_l23u.append(l4_combinations[2].shape[0]/layer_4_untuned_out)
        l4u_l23t.append(l4_combinations[3].shape[0]/layer_4_tuned_out)

        #L2/3 -> L2/3
        #Normalising constants
        layer_23_tuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 1)
                                & (v1_connections_samp['pre_layer'] == 'L2/3')].shape[0]

        layer_23_untuned_out = v1_connections_samp[(v1_connections_samp['post_tuned'] == 0)
                                & (v1_connections_samp['pre_layer'] == 'L2/3')].shape[0]

        #Probabilities
        l23t_l23t.append(l23_combinations[0].shape[0]/layer_23_tuned_out)
        l23t_l23u.append(l23_combinations[1].shape[0]/layer_23_untuned_out)
        l23u_l23u.append(l23_combinations[2].shape[0]/layer_23_untuned_out)
        l23u_l23t.append(l23_combinations[3].shape[0]/layer_23_tuned_out)


    l4_tuning_connp = [l4t_l23t, l4t_l23u, l4u_l23t, l4u_l23u]
    l23_tuning_connp = [l23t_l23t, l23t_l23u, l23u_l23t, l23u_l23u]
    return l23_tuning_connp, l4_tuning_connp


def prepare_c2(v1_connections):

    #Identify tuned neurons
    tuned_neurons = v1_connections[(v1_connections['post_type']!= 'not_selective') & 
                                   (v1_connections['pre_type']!= 'not_selective')]

    #Extract bootstrapped stats
    l23_boots = bootstrap_conn_prob(tuned_neurons,pre_layer='L2/3')
    l4_boots = bootstrap_conn_prob(tuned_neurons, pre_layer='L4')

    return l23_boots, l4_boots

def prepare_d2(v1_connections):

    #Extract tuning combinations
    l4_combinations, l23_combinations = tuning_segmenter(v1_connections)

    #L4 -> L2/3
    l4t_l23t_strengths = np.array(l4_combinations[0]['size'])
    l4t_l23u_strengths = np.array(l4_combinations[1]['size'])
    l4u_l23u_strengths =np.array(l4_combinations[2]['size'])
    l4u_l23t_strengths =np.array(l4_combinations[3]['size'])

    #L2/3 -> L2/3
    l23t_l23t_strengths = np.array(l23_combinations[0]['size'])
    l23t_l23u_strengths = np.array(l23_combinations[1]['size'])
    l23u_l23u_strengths = np.array(l23_combinations[2]['size'])
    l23u_l23t_strengths = np.array(l23_combinations[3]['size'])
    
    l4_comb_strengths = [l4t_l23t_strengths,l4t_l23u_strengths,l4u_l23t_strengths, l4u_l23u_strengths]
    l23_comb_strengths = [l23t_l23t_strengths,l23t_l23u_strengths,l23u_l23t_strengths, l23u_l23u_strengths]

    return l23_comb_strengths, l4_comb_strengths

def prepare_e2(v1_connections, angles_list, bins = 20):

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
    
    ch1, b1 = cumul_dist(sub1l4,20)

    ch2, b2 = cumul_dist(sub2l4,20)

    ch3, b3 = cumul_dist(sub1l23,20)

    ch4, b4 = cumul_dist(sub2l23,20)

    return [[ch1, b1], [ch2, b2], [ch3, b3], [ch4, b4]]



def prepare_b3(v1_connections, proofread_input_n):

    tuned_outputs = v1_connections[(v1_connections['post_type']!= 'not_selective') 
                                   & (v1_connections['pre_type']!= 'not_selective')]
    l23_t = tuned_outputs[tuned_outputs['pre_layer'] == 'L2/3']
    l23_t = l23_t.copy()
    l4_t = tuned_outputs[tuned_outputs['pre_layer'] == 'L4']
    l4_t = l4_t.copy()
    
    #Calculate aggregated currents
    t23_grouped, _, _= all_inputs_aggregator2(l23_t, 'shifted_current', 'new_dirs', grouping = 'mean' )#, dir_range='half')
    t4_grouped, _, _= all_inputs_aggregator2(l4_t, 'shifted_current', 'new_dirs', grouping = 'mean' )#, dir_range='half')

    # calculate number of L2/3 inputs and number of L4 inputs
    ave_l23 = np.mean(proofread_input_n[proofread_input_n['cortex_layer'] == 'L2/3']['n_connections'])
    ave_l4 = np.mean(proofread_input_n[proofread_input_n['cortex_layer'] == 'L4']['n_connections'])
    tot = ave_l23+ave_l4

    #Proportions of inputs from each layer
    propl23 = ave_l23/tot
    propl4 = ave_l4/tot

    return t23_grouped, propl23, t4_grouped, propl4


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