import numpy as np
from scipy import stats
import pandas as pd

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


        

           
        '''
        print(fraction_of_neurons_type,fraction_of_neurons_tuning)
        print(pref_ori)
        print(fraction_of_neurons_pref_ori[:,0])
        print(np.sum(fraction_of_neurons_pref_ori[:,0]))
        '''
        
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


            '''
            print(label_post)
            print(label_pre)
            print(prob_conn[0], prob_conn[1])
            print(np.mean(sampled_J), stats.sem(sampled_J))
            print()
            '''
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


def sample_neurons_with_tuning(Frac_,N, Labels):
    #Generate a sample from inferred probabilities
    # Initialize an empty list to store the sampled labels
    area_list = []
    layer_list = []
    cell_type_list = []
    tuning_type_list=[]
    pref_ori_list=[]
    for label_info in Labels:
        #print(label_info)
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
            
        if (tuning_type=='selective')&(fraction_of_neurons_tuning>0):
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
    neurons['pt_root_id']=np.arange(len(pref_ori_list))
    neurons['pt_root_id']=np.arange(len(pref_ori_list))
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

        mask_cell_post = (neurons['cell_type'] == cell_type_post) & \
                         (neurons['layer'] == layer_post)
        mask_cell_post = mask_cell_post&(neurons['tuning_type'] == tuning_type_post)
        post_id_list=neurons['pt_root_id'][mask_cell_post].values
        possible_ori_post=np.unique(neurons['pref_ori'][mask_cell_post])


        for label_pre in Labels:
            label_key_pre = tuple(sorted(label_pre.items()))
            area_pre = label_pre['area']
            layer_pre = label_pre['layer']
            cell_type_pre = label_pre['cell_type']
            tuning_type_pre = label_pre['tuning_type']
    
            mask_cell_pre = (neurons['cell_type'] == cell_type_pre) & \
                            (neurons['layer'] == layer_pre)
            mask_cell_pre = mask_cell_pre & (neurons['tuning_type'] == tuning_type_pre)
            mask_cell_pre=mask_cell_pre&(neurons['axon_proof']!='non')
            pre_id_list=neurons['pt_root_id'][mask_cell_pre].values
            possible_ori_pre=np.unique(neurons['pref_ori'][mask_cell_pre])
            

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
                        
                        post_id_list_with_ori=neurons['pt_root_id'][mask_cell_post_with_ori].values
                        pre_id_list_with_ori=neurons['pt_root_id'][mask_cell_pre_with_ori].values

                        dist=dist_ell(pref_ori_post,pref_ori_pre,Lmax)
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

    sampled_connections = pd.DataFrame({'pre_pt_root_id':pre_pt_root_id_list,
                   'post_pt_root_id':post_pt_root_id_list,
                   'syn_volume':syn_volume_list,
                    })
    #remove self-connections
    sampled_connections=sampled_connections[sampled_connections['pre_pt_root_id']!= sampled_connections['post_pt_root_id']]

    return sampled_connections
