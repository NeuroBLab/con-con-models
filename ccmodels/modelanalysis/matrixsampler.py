import numpy as np
import pandas as pd

#TODO this was calling functions file in victor
import ccmodels.modelanalysis.functions_new as fun

import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as utl
import ccmodels.dataanalysis.processedloader as loader

#TODO this is a WIP sampler made from Alessandro's code

# load files
def get_fractions(unit_table, connections_table, Labels, local_connectivity=True):

    Frac_tables_data=fun.measure_fractions_of_neurons(unit_table,Labels)
    Conn_stat_measured_on_data=fun.measure_connection_stats(connections_table, unit_table,Labels)

    Conn_stat_measured_on_data=fun.modify_Conn_stat_measured_on_data(Conn_stat_measured_on_data, local_connectivity)

    return Frac_tables_data, Conn_stat_measured_on_data 

#TODO this has to be called from the parent script between the previous two functions
#N=1000, target_K_EE=200
#scaling_prob=fun.Compute_scaling_factor_for_target_K_EE(connections_table, unit_table,target_K_EE,N)

def generate_functions(scaling_prob, Frac_tables_data, Conn_stat_measured_on_data, Labels, N=1000):

    # sample and measure stats on sampled quantities
    neurons_sampled=sample_neurons_with_tuning(Frac_tables_data,N,Labels)
    #TODO frac_tables for the moment is useless
    #Frac_tables_sampled=fun.measure_fractions_of_neurons(neurons_sampled,Labels)
    sampled_connections=sample_connections(Conn_stat_measured_on_data,neurons_sampled,scaling_prob,Labels)

    #return neurons_sampled,Frac_tables_sampled, sampled_connections
    return neurons_sampled, sampled_connections


def generate_conn_matrix(neurons_sampled, sampled_connections, J, g):
    # Initialize QJ array with zeros
    QJ = np.zeros((len(neurons_sampled), len(neurons_sampled)))

    # Get the necessary data from sampled_connections
    pre_pt_root_ids = sampled_connections['pre_id']
    post_pt_root_ids = sampled_connections['post_id']
    syn_volumes = sampled_connections['syn_volume']

    # Assign synapse volumes to QJ array based on pre and post synaptic root IDs and scale by factor J
    QJ[post_pt_root_ids, pre_pt_root_ids] = J*syn_volumes

    # scale inhibitory connections and make them negative
    l23_sampled = fl.filter_neurons(neurons_sampled, layer='L23')
    l23E = fl.filter_neurons(l23_sampled, cell_type='exc')
    l23I = fl.filter_neurons(l23_sampled, cell_type='inh')
    num_L23_neurons_E = len(l23E)
    num_L23_neurons_I = len(l23I)


    QJ[:, num_L23_neurons_E:(num_L23_neurons_E+num_L23_neurons_I)] *= -g 

    # Remove post synaptic neurons in L4
    num_L23_neurons = len(l23_sampled)
    num_L4_neurons = len(fl.filter_neurons(neurons_sampled, layer='L4', cell_type='exc'))
    
    QJ = QJ[:num_L23_neurons, :]

    return QJ, num_L23_neurons_E, num_L23_neurons_I, num_L4_neurons 


def sample_neurons_with_tuning(Frac_,N, Labels):
    #Generate a sample from inferred probabilities
    # Initialize an empty list to store the sampled labels
    area_list = []
    layer_list = []
    cell_type_list = []
    tuning_type_list=[]
    pref_ori_list=[]

    for label_info in Labels:
        area = label_info['area']
        layer = label_info['layer']
        cell_type = label_info['cell_type']
        tuning_type = label_info['tuning_type']

        # Convert label_info dictionary into a tuple
        label_key = tuple(sorted(label_info.items()))
        
        # Access corresponding probabilities/possible orientations using the tuple as key
        frac_info = Frac_[label_key]
    
        fraction_of_neurons_type=Frac_[label_key]['fraction_of_neurons_type']
        fraction_of_neurons_tuning=Frac_[label_key]['fraction_of_neurons_tuning']
        pref_ori=Frac_[label_key]['pref_ori']
        fraction_of_neurons_pref_ori=Frac_[label_key]['fraction_of_neurons_pref_ori'][:,0]
        
        frac_type_tuning=fraction_of_neurons_type*fraction_of_neurons_tuning

        if tuning_type=='not_selective':
            #no neuro in the populaiton is selective, case of I neurons
            N_type_tuning=int(frac_type_tuning * N)
            tuning_type_list+=([tuning_type]*N_type_tuning)
            pref_ori_list+=([np.nan]*N_type_tuning)
            area_list+=[area]*N_type_tuning
            layer_list+=([layer]*N_type_tuning)
            cell_type_list+=([cell_type]*N_type_tuning)

        elif (tuning_type=='selective')&(fraction_of_neurons_tuning>0):
            #no neuro in the populaiton is selective, case of I neurons
            count_N_type_tuning_ori=0
            for idx_ori in range(len(pref_ori)):  
                N_type_tuning_ori=int(frac_type_tuning*fraction_of_neurons_pref_ori[idx_ori] * N)
                count_N_type_tuning_ori+=N_type_tuning_ori            
                tuning_type_list+=([tuning_type]*N_type_tuning_ori)
                pref_ori_list+=([pref_ori[idx_ori]]*N_type_tuning_ori)
                
                area_list+=[area]*N_type_tuning_ori
                layer_list+=([layer]*N_type_tuning_ori)
                cell_type_list+=([cell_type]*N_type_tuning_ori)
        
    neurons = pd.DataFrame({'area':area_list,
                       'layer':layer_list,
                       'cell_type':cell_type_list,
                       'tuning_type':tuning_type_list,
                       'pref_ori':pref_ori_list,})

    neurons['id']=np.arange(len(pref_ori_list))
    neurons['id']=np.arange(len(pref_ori_list))
    neurons['axon_proof']='extended'
    return neurons

def sample_connections(Conn_stat,neurons,scaling_prob,Labels):
    # ATT the code is written to remove self-connections
    post_pt_root_id_list = []
    pre_pt_root_id_list = []
    syn_volume_list=[]

    for label_post in Labels[0:4]:
        label_key_post = tuple(sorted(label_post.items()))
        area_post = label_post['area']
        layer_post = label_post['layer']
        cell_type_post = label_post['cell_type']
        tuning_type_post = label_post['tuning_type']

        mask_cell_post = (neurons['cell_type'] == cell_type_post) & (neurons['layer'] == layer_post) &(neurons['tuning_type'] == tuning_type_post)
        post_id_list=neurons.loc[mask_cell_post, 'id'].values
        #possible_ori_post=np.unique(neurons['pref_ori'][mask_cell_post])
        #!
        possible_ori_post=neurons.loc[mask_cell_post, 'pref_ori'].unique()


        for label_pre in Labels:
            label_key_pre = tuple(sorted(label_pre.items()))
            area_pre = label_pre['area']
            layer_pre = label_pre['layer']
            cell_type_pre = label_pre['cell_type']
            tuning_type_pre = label_pre['tuning_type']
    
            mask_cell_pre = (neurons['cell_type'] == cell_type_pre) & (neurons['layer'] == layer_pre) & (neurons['tuning_type'] == tuning_type_pre) &(neurons['axon_proof']!='non')
            pre_id_list=neurons.loc[mask_cell_pre, 'id'].values
            #possible_ori_pre=np.unique(neurons['pref_ori'][mask_cell_pre])
            #!
            possible_ori_pre = neurons.loc[mask_cell_pre, 'pref_ori'].unique()
            

            prob_conn=Conn_stat[label_key_post,label_key_pre,]['prob_conn'].copy()
            sampled_J=Conn_stat[label_key_post,label_key_pre,]['sampled_J'].copy()
            dist_values=Conn_stat[label_key_post,label_key_pre,]['dist_values'].copy()
            prob_conn_vs_dist=Conn_stat[label_key_post,label_key_pre,]['prob_conn_vs_dist'].copy()
            sampled_J_vs_dist=Conn_stat[label_key_post,label_key_pre,]['sampled_J_vs_dist'].copy()

            prob_conn[0]=scaling_prob*prob_conn[0]
            prob_conn_vs_dist[:,0]=scaling_prob*prob_conn_vs_dist[:,0]

                
            if (tuning_type_post=='selective')&(tuning_type_pre=='selective')&(len(possible_ori_post)>0)&(len(possible_ori_pre)>0):
                Lmax=np.max(possible_ori_pre)+1
                
                for pref_ori_post in possible_ori_post:
                    mask_cell_post_with_ori=(neurons['pref_ori']==pref_ori_post)&mask_cell_post
                    for pref_ori_pre in possible_ori_pre:
                        mask_cell_pre_with_ori=(neurons['pref_ori']==pref_ori_pre)&mask_cell_pre
                        
                        post_id_list_with_ori=neurons['id'][mask_cell_post_with_ori].values
                        pre_id_list_with_ori=neurons['id'][mask_cell_pre_with_ori].values

                        dist=fun.dist_ell(pref_ori_post,pref_ori_pre,Lmax)
                        prob_conn_to_use=prob_conn_vs_dist[prob_conn_vs_dist[:,2]==dist,0][0]
                        sampled_J_to_use=sampled_J_vs_dist[sampled_J_vs_dist[:,1]==dist,0]

                        
                        Q=np.random.rand(len(post_id_list_with_ori),len(pre_id_list_with_ori))<prob_conn_to_use
                        indices=np.where(Q==1)
                        post_pt_root_id_list+=post_id_list_with_ori[indices[0]].tolist()
                        pre_pt_root_id_list+=pre_id_list_with_ori[indices[1]].tolist()
                        syn_volume_list+=np.random.choice(sampled_J_to_use, size=len(indices[0]), replace=True).tolist()
                        
                
            else:
                prob_conn_to_use=prob_conn[0]
                sampled_J_to_use=sampled_J
                Q=np.random.rand(len(post_id_list),len(pre_id_list))<prob_conn_to_use
                indices=np.where(Q==1)
                post_pt_root_id_list+=post_id_list[indices[0]].tolist()
                pre_pt_root_id_list+=pre_id_list[indices[1]].tolist()
                syn_volume_list+=np.random.choice(sampled_J_to_use, size=len(indices[0]), replace=True).tolist()

    sampled_connections = pd.DataFrame({'pre_id':pre_pt_root_id_list,
                   'post_id':post_pt_root_id_list,
                   'syn_volume':syn_volume_list,
                    })

    #remove self-connections
    sampled_connections=sampled_connections[sampled_connections['pre_id']!= sampled_connections['post_id']]

    return sampled_connections

def sample_L4_rates(units, activity, units_sample, nangles=16, orionly=False):

    #Get neurons and their activity in L4
    neurons_L4 = fl.filter_neurons(units, layer='L4', tuning='matched')

    #Now, get the neurons sampled from the synthetic model
    neurons_L4_sample = fl.filter_neurons(units_sample, layer='L4', tuning='matched')

    #Number of neurons in each case
    n = len(neurons_L4)
    n_sample = len(neurons_L4_sample)

    #Set the rates of untuned neurons to their average
    act_matrix = utl.get_untuned_rate(units, activity)

    #Get the submatrix of rates in L4
    act_matrix = act_matrix[neurons_L4['id'], :]

    #Shift all neurons so the largest rate is centered at 0
    act_matrix = utl.shift_multi(act_matrix, neurons_L4['pref_ori'])

    #Number of tuned neurons in data and synthetic tables
    n_tuned_data   = len(fl.filter_neurons(neurons_L4, tuning='tuned'))
    n_tuned_sample = len(fl.filter_neurons(neurons_L4_sample, tuning='tuned'))

    #Randomly sample neurons in the tuned and untuned part, respectively. Then put all together
    idx_tuned = np.random.choice(np.arange(0, n_tuned_data), size=n_tuned_sample, replace=True)
    idx_untuned = np.random.choice(np.arange(n_tuned_data, n), size=n_sample - n_tuned_sample, replace=True)
    idx_selected = np.concatenate([idx_tuned, idx_untuned])

    #Fill the matrix with the sampled ids
    act_matrix = act_matrix[idx_selected, :]

    #The tuned ones have to to be moved back to their pref oris
    act_matrix[:n_tuned_sample] = utl.shift_multi(act_matrix[:n_tuned_sample], -neurons_L4_sample.iloc[:n_tuned_sample, 4])

    return act_matrix