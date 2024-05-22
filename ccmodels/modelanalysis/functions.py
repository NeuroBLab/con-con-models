import numpy as np
from scipy import stats
import pandas as pd

#TODO this is a temporary file. A lot of it should go into statistics extraction, and we actually a function that already does what this code does
#The rest of it goes directly to the matrixsampler

def Compute_normalization_factor_for_synaptic_volume(connections,neurons):

    # compute avereage connection strength between connected pairs of exc L23 neruons
    mask_selective=(neurons['tuning_type']=='selective')
    mask_not_selective=(neurons['tuning_type']=='not_selective')
    mask_matched=mask_selective|mask_not_selective

    mask_cell_post = (neurons['cell_type'] == 'exc') & (neurons['layer'] == 'L23')&mask_matched # I add matched because in the estimation of J_s for E I used selective or not, which is valid for matched
    mask_cell_pre = mask_cell_post&(neurons['axon_proof'] != 'non')

    post_id_list = neurons.loc[mask_cell_post, 'pt_root_id'].values
    pre_id_list = neurons.loc[mask_cell_pre, 'pt_root_id'].values
    
    mask_conn = connections['pre_pt_root_id'].isin(pre_id_list) & \
                connections['post_pt_root_id'].isin(post_id_list)

    sampled_J_EE = connections[mask_conn]['syn_volume'].values.copy()
    norm_V=np.mean(sampled_J_EE)
    return norm_V


def Compute_scaling_factor_for_target_K_EE(connections,neurons,target_K_EE,new_N):
    # connections probability are computed looking at matched cells. 
    # I have to do the same here to get be consistent
    mask_selective=(neurons['tuning_type']=='selective')
    mask_not_selective=(neurons['tuning_type']=='not_selective')
    mask_matched=mask_selective|mask_not_selective
    mask_cell_post = (neurons['cell_type'] == 'exc') & (neurons['layer'] == 'L23')&mask_matched
    mask_cell_pre = mask_cell_post&(neurons['axon_proof'] != 'non')
    post_id_list = neurons.loc[mask_cell_post, 'pt_root_id'].values
    pre_id_list = neurons.loc[mask_cell_pre, 'pt_root_id'].values
    norm = len(post_id_list)*(len(pre_id_list)-1)
    mask_conn = connections['pre_pt_root_id'].isin(pre_id_list) &  connections['post_pt_root_id'].isin(post_id_list)
    count=len(connections[mask_conn])
    P_EE=count/norm

    mask_cell=(neurons['cell_type'] == 'exc') & (neurons['layer'] == 'L23')
    N_E=len(neurons[mask_cell])
    K_EE_from_data=P_EE*N_E
    scaling_prob=target_K_EE/K_EE_from_data*len(neurons)/new_N
    return scaling_prob


def measure_fractions_of_neurons(neurons,Labels):
    # function to measure fraction of neurons with given characteristics

    mask_selective=(neurons['tuning_type']=='selective')
    mask_not_selective=(neurons['tuning_type']=='not_selective')
    mask_matched=mask_selective|mask_not_selective

    Frac_tables = {}
    # number of neurons in the dataset, used for normalization in the stimate of probability  of neurons belonging to a label
    norm = len(neurons)
    for label_info in Labels:
        
        area = label_info['area']
        layer = label_info['layer']
        cell_type = label_info['cell_type']
        tuning_type = label_info['tuning_type']
    
        mask_cell_type = (neurons['cell_type'] == cell_type) & (neurons['layer'] == layer)
        count = len(neurons[mask_cell_type])
        fraction_of_neurons_type =count / norm

        mask_cell_type_matched=mask_cell_type&mask_matched
        norm_matched=len(neurons[mask_cell_type_matched])
        
        if norm_matched==0:
            # no neuron of that type is functionally matched
            if tuning_type=='not_selective':
                fraction_of_neurons_tuning =1
                pref_ori=np.nan*np.ones(1)
                fraction_of_neurons_pref_ori = np.zeros((len(pref_ori),2))
                fraction_of_neurons_pref_ori[0,0]=1

            if tuning_type=='selective':
                fraction_of_neurons_tuning =0
                pref_ori=np.nan*np.ones(1)
                fraction_of_neurons_pref_ori = np.zeros((len(pref_ori),2))
                fraction_of_neurons_pref_ori[0,0]=1

        if norm_matched>0:
            # some neurons of that type are functionally matched
            if tuning_type=='not_selective':
                mask_cell_type_not_selective=mask_cell_type&mask_not_selective
                count = len(neurons[mask_cell_type_not_selective])
                fraction_of_neurons_tuning =count/norm_matched
                pref_ori=np.nan*np.ones(1)
                fraction_of_neurons_pref_ori = np.zeros((len(pref_ori),2))
                fraction_of_neurons_pref_ori[0,0]=1

            if tuning_type=='selective':
                mask_cell_type_selective=mask_cell_type&mask_selective
                count = len(neurons[mask_cell_type_selective])
                fraction_of_neurons_tuning =count/norm_matched

                
                #compute fraction of selective neurons with given preferred orientaiton 
                norm_selective=count
                pref_ori = np.unique(neurons.loc[mask_cell_type_selective, 'pref_ori'])
                
                fraction_of_neurons_pref_ori = np.zeros((len(pref_ori),2))
                for idx in range(len(pref_ori)):
                    # not selective have with pref_ori nan
                    mask_pref_ori = (neurons['pref_ori'] == pref_ori[idx]) & mask_cell_type_selective
                    count = len(neurons[mask_pref_ori])
                    prob_conn=count / norm_selective
                    fraction_of_neurons_pref_ori[idx,0] = prob_conn
                    fraction_of_neurons_pref_ori[idx,1] = np.sqrt(prob_conn * (1 - prob_conn) / norm_selective)


        # Convert label_info dictionary into a tuple for use as a key
        label_key = tuple(sorted(label_info.items()))
    
        # Store probabilities with labels as keys in the dictionary
        Frac_tables[label_key] = {'fraction_of_neurons_type': fraction_of_neurons_type,
                                  'fraction_of_neurons_tuning': fraction_of_neurons_tuning,
                                 'pref_ori': pref_ori,
                                 'fraction_of_neurons_pref_ori': fraction_of_neurons_pref_ori}
    return Frac_tables

# Auxiliary functions to measure connection stats

def dist_ell(l1,l2,Lmax):
    # Compute the distance between neurons with labels l1 and l2. 
    # Assumes labels are circular variables, could be generalized to other cases
    return np.min((np.abs(l1-l2),Lmax-np.abs(l1-l2)))

def measure_P_of_dist_ell(post_neurons,pre_neurons, connections):
    # post_neurons, pre_neurons and connections should be pd DataFrames  

    L, _ = np.unique((pre_neurons['pref_ori'].values), return_counts=True) # Should be the same for pre and post
    Lmax=np.max(L)+1
    
    sampled_J_vs_d = np.zeros((1,2))
    for l_post in L:
        post_id_list=post_neurons['pt_root_id'][post_neurons['pref_ori'] == l_post].values
        for l_pre in L:
            pre_id_list=pre_neurons['pt_root_id'][pre_neurons['pref_ori'] == l_pre].values
            mask=connections['pre_pt_root_id'].isin(pre_id_list)& connections['post_pt_root_id'].isin(post_id_list)

            d = dist_ell(l_post, l_pre, Lmax)
            data = connections[mask]['syn_volume'].values
            if l_post!=l_pre:
                pippo=np.zeros((len(post_id_list)*len(pre_id_list),2))
            if l_post==l_pre:
                pippo=np.zeros((len(post_id_list)*(len(pre_id_list)-1),2))
            pippo[:,1]=d
            pippo[0:len(data),0]=data

            sampled_J_vs_d=np.concatenate((sampled_J_vs_d,pippo))
    sampled_J_vs_d = sampled_J_vs_d[1:]

    dist_values = np.unique(sampled_J_vs_d[:, 1])
    prob_conn_vs_dist = np.zeros((len(dist_values),3))
    
    for i, d in enumerate(dist_values):
        pippo=sampled_J_vs_d[sampled_J_vs_d[:, 1] == d, 0].copy()
        pippo[pippo>0]=1
        prob_conn_vs_dist[i,:] = np.mean(pippo), stats.sem(pippo),d
    
    return dist_values, prob_conn_vs_dist,sampled_J_vs_d[sampled_J_vs_d[:,0]>0,:]

def measure_connection_stats(connections,neurons,Labels):

    mask_selective=(neurons['tuning_type']=='selective')
    mask_not_selective=(neurons['tuning_type']=='not_selective')
    mask_matched=mask_selective|mask_not_selective

    Conn_table_with_tuning_stat = {}


    for label_post in Labels:
        label_key_post = tuple(sorted(label_post.items()))
        area_post = label_post['area']
        layer_post = label_post['layer']
        cell_type_post = label_post['cell_type']
        tuning_type_post = label_post['tuning_type']

        mask_cell_type_post = (neurons['cell_type'] == cell_type_post) & \
                         (neurons['layer'] == layer_post)

        if len(neurons[mask_cell_type_post&mask_matched])>0:
            if tuning_type_post=='selective':
                mask_cell_type_post_tuning = mask_cell_type_post&mask_selective   
            if tuning_type_post=='not_selective':  
                mask_cell_type_post_tuning = mask_cell_type_post&mask_not_selective   
        
        if len(neurons[mask_cell_type_post&mask_matched])==0:
            # there are no neurons matched of that type, 
            # I consider all neurons as not selective and that is it
            if tuning_type_post=='selective':
                mask_cell_type_post_tuning = mask_cell_type_post&mask_matched #(False for everyone) 
            if tuning_type_post=='not_selective':
                mask_cell_type_post_tuning = mask_cell_type_post #(True for everyone) 
            
        post_id_list = neurons.loc[mask_cell_type_post_tuning, 'pt_root_id'].values
        
        for label_pre in Labels:
            label_key_pre = tuple(sorted(label_pre.items()))
            area_pre = label_pre['area']
            layer_pre = label_pre['layer']
            cell_type_pre = label_pre['cell_type']
            tuning_type_pre = label_pre['tuning_type']

            mask_cell_type_pre = (neurons['cell_type'] == cell_type_pre) & \
                            (neurons['layer'] == layer_pre) & \
                            (neurons['axon_proof'] != 'non')


            if len(neurons[mask_cell_type_pre&mask_matched])>0:
                if tuning_type_pre=='selective':
                    mask_cell_type_pre_tuning = mask_cell_type_pre&mask_selective   
                if tuning_type_pre=='not_selective':  
                    mask_cell_type_pre_tuning = mask_cell_type_pre&mask_not_selective   
            
            if len(neurons[mask_cell_type_pre&mask_matched])==0:
                # there are no neurons matched of that type, 
                # I consider all neurons as not selective and that is it
                if tuning_type_pre=='selective':
                    mask_cell_type_pre_tuning = mask_cell_type_pre&mask_matched #(False for everyone) 
                if tuning_type_pre=='not_selective':
                    mask_cell_type_pre_tuning = mask_cell_type_pre #(True for everyone) 
                
            pre_id_list = neurons.loc[mask_cell_type_pre_tuning, 'pt_root_id'].values

            # Focus on connections between neurons of specific pre and post tuning
            mask_conn = connections['pre_pt_root_id'].isin(pre_id_list) & \
                        connections['post_pt_root_id'].isin(post_id_list)

            # First I compute connection probability based on the difference in cell type and tuning
            # I do not consider preferred orientation here
            sampled_J = connections[mask_conn]['syn_volume'].values
            conn_obs=len(sampled_J)

            # I neglect the -1 in len(pre_id_list) for symmetric connections
            norm = len(post_id_list)*len(pre_id_list)
            prob_conn=np.zeros(2)
            if norm>0:
                prob_conn[0]=conn_obs/norm
                prob_conn[1]=np.sqrt(prob_conn[0]*(1-prob_conn[0])/norm)


            # For  selective post and pre I differentitate for prob as a funciton of distance in tuning
            dist_values=np.nan*np.ones(1)
            prob_conn_vs_dist=np.nan*np.ones((1,3))
            sampled_J_vs_d=np.nan*np.ones((1,2))
            if (tuning_type_post=='selective')&(tuning_type_pre=='selective')&(len(post_id_list)>0)&(len(pre_id_list)>0):
                dist_values,prob_conn_vs_dist,sampled_J_vs_dist=measure_P_of_dist_ell(neurons[mask_cell_type_post_tuning],
                                                                                       neurons[mask_cell_type_pre_tuning], 
                                                                  connections[mask_conn],)
                

            Conn_table_with_tuning_stat[label_key_post,label_key_pre,] = {
                            'prob_conn':prob_conn,
                            'sampled_J':sampled_J,
                            'dist_values':dist_values,
                            'prob_conn_vs_dist':prob_conn_vs_dist,
                            'sampled_J_vs_dist':sampled_J_vs_dist,
                            }
    return Conn_table_with_tuning_stat


