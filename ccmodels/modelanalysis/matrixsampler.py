import numpy as np
import pandas as pd

import ccmodels.dataanalysis.filters as fl
import ccmodels.dataanalysis.utils as utl
import ccmodels.dataanalysis.statistics_extraction as ste

import ccmodels.utils.angleutils as au

def sample_L4_rates(units, activity, units_sample, mode='normal'): 
    #Get neurons and their activity in L4
    neurons_L4 = fl.filter_neurons(units, layer='L4', tuning='matched')

    #Now, get the neurons sampled from the synthetic model
    neurons_L4_sample = fl.filter_neurons(units_sample, layer='L4', tuning='matched').reset_index()

    #Number of neurons in each case
    n = len(neurons_L4)
    n_sample = len(neurons_L4_sample)

    #Set the rates of untuned neurons to their average
    act_matrix = utl.get_untuned_rate(units, activity)

    #Get the submatrix of rates in L4
    act_matrix = act_matrix[neurons_L4['id'], :]

    if mode == 'random':
        #In the random case, it suffices to just sample from the system's statistics. 
        #Doing that, fractions of tuned/untuned and each pref ori is respected.
        idx_selected = np.random.choice(np.arange(0, n), size=n_sample, replace=True)
        act_matrix = act_matrix[idx_selected, :] 
    else:
        #In the normal case, each neuron has a predefined orientation that we must match.
        #So we have to sample from the system, shifting at zero, and then shift again to each neuron's ori

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
        act_matrix[:n_tuned_sample] = utl.shift_multi(act_matrix[:n_tuned_sample], -neurons_L4_sample.loc[:n_tuned_sample-1, 'pref_ori'])

    return act_matrix

def compute_scaling_factor_kEE(neurons, connections, target_k_EE,new_N):

    #Filter the exc neuron L23 and get pre and postsynaptic matched neurons, ensuring
    #that presynaptic are proofread
    e_neurons = fl.filter_neurons(neurons, layer='L23', cell_type='exc')
    post_id_list = fl.filter_neurons(e_neurons, tuning='matched') 
    #post_id_list = e_neurons
    pre_id_list = fl.filter_neurons(post_id_list, proofread='minimum')
    #pre_id_list = fl.filter_neurons(post_id_list, proofread='ax_clean')
    conn_filtered = fl.synapses_by_id(connections, pre_ids=pre_id_list['id'], post_ids=post_id_list['id'], who='both')

    #Possible amount of links
    norm = (len(post_id_list)-1)*len(pre_id_list)
    
    #Use these to get the probability of connections and number of neurons
    count=len(conn_filtered)
    p_EE = count/norm
    N_E = len(e_neurons)

    #Average connectivity in data
    k_EE_data = p_EE * N_E

    #Get the factor
    return (target_k_EE/k_EE_data) * (len(neurons)/new_N)


def sample_matrix(units, connections, k_ee, N, J, g, prepath='data', mode='normal', cos_a=[0.9807840158851815, 1.051784991962299], cos_b=[0.17446353427026207, 0.15346752188086193]):
    #Get the scaling for a correct definition of k_ee
    #scaling_prob=fun.Compute_scaling_factor_for_target_K_EE(connections, units, k_ee, N)
    #scaling_prob=compute_scaling_factor_kEE(units, connections, k_ee, N)

    #Get the fraction of total size that each population has as a pandas Series
    #This was previously estimated from data
    fractions = pd.read_csv(f"{prepath}/model/fractions_populations.csv", index_col='Population')
    fractions = fractions.squeeze()

    #Read the connection probabilities between each paper of populations 
    #This was previously estimated from data
    if mode=='normal' or 'tuned' in mode:
        ptable = pd.read_csv(f"{prepath}/model/prob_connectomics_cleanaxons.csv", index_col="Unnamed: 0") 
        av_prob = ptable.loc['E', 'E']
        ptable = pd.read_csv(f"{prepath}/model/prob_cleanaxons.csv", index_col="Population") 
        #Scale to get our desired k_ee. In this case the avg E-E probability is taken from connectomics
        scaling_prob = k_ee / (N * fractions['E'] * av_prob) 
        ptable *= scaling_prob
    #Unstructured network from connectomics
    elif mode == 'random':
        ptable = pd.read_csv(f"{prepath}/model/prob_connectomics_cleanaxons.csv", index_col="Unnamed: 0") 
        #Scale to get our desired k_ee 
        scaling_prob = k_ee / (N*fractions['E']*ptable.loc['E', 'E']) 
        ptable *= scaling_prob 
    elif mode=='cosine':
        #Computing the scaling to get our desired k_ee 
        ptable_con = pd.read_csv(f"{prepath}/model/prob_connectomics_cleanaxons.csv", index_col="Unnamed: 0") 
        scaling_prob = k_ee / (N*fractions['E']*ptable_con.loc['E', 'E']) 

        #Compute the modulation we will add to our connectomic probability
        nangles = 8
        theta =  np.arange(nangles)
        cosine = np.cos(2*np.pi*theta/nangles)

        #Observe that modulated[0] > 1 and modulated[pi/2] < 1 to keep the average correct
        modulated_EE = cos_a[0] + cos_b[0] * cosine
        modulated_EX = cos_a[1] + cos_b[1] * cosine

        #Define our new table from scratch, creating the colums
        ETcols = [f'ET_{i}' for i in range(nangles)]
        XTcols = [f'XT_{i}' for i in range(nangles)]
        columns = ETcols + ['I'] + XTcols  #+ ['XU']
        ptable = pd.DataFrame(columns= columns)


        #Then we fill all the postsynaptic rows
        for i in theta:
            #Create the rows
            ptable.loc[f"ET_{i}", :] = 0. 

            #Fill the values for each set of columns
            ptable.loc[f"ET_{i}", ETcols] = np.roll(ptable_con.loc['E', 'E'] * modulated_EE, i)
            ptable.loc[f"ET_{i}", "I"] = ptable_con.loc['E', 'I'] 
            ptable.loc[f"ET_{i}", XTcols] = np.roll(ptable_con.loc['E', 'X'] * modulated_EX, i)
        
        #Do the same with inhibitory neurons. There's no L4 postsynaptic so this is all.
        ptable.loc['I', :] = 0
        ptable.loc["I", ETcols] = ptable_con.loc['I', 'E']
        ptable.loc["I", 'I'] = ptable_con.loc['I', 'I']
        ptable.loc["I", XTcols] = ptable_con.loc['I', 'X']

        #free intermediate memory
        del ptable_con

        ptable *= scaling_prob

        #Redistribute the fraction of untuned neurons equally between tuned ones
        labels = [f"ET_{i}" for i in range(8)]
        fractions[labels] += fractions['EU'] * fractions[labels] / fractions['ET']
        labels = [f"XT_{i}" for i in range(8)]
        fractions[labels] += fractions['XU'] * fractions[labels] / fractions['XT']

        #Add the total number of untuned to the total number of tuned now
        fractions["ET"] += fractions['EU']
        fractions["XT"] += fractions['XU']



    #Get the number of neurons we have, from the fractions
    n_neurons = np.array([fractions['E'], fractions['I'], fractions['X']]) * N
    n_neurons = np.round(n_neurons).astype(int)

    #In this mode we do not work with untuned neurons
    if 'tuned' in mode:

        #Redistribute the fraction of untuned neurons equally between tuned ones
        labels = [f"ET_{i}" for i in range(8)]
        fractions[labels] += fractions['EU'] * fractions[labels] / fractions['ET']
        labels = [f"XT_{i}" for i in range(8)]
        fractions[labels] += fractions['XU'] * fractions[labels] / fractions['XT']

        #Add the total number of untuned to the total number of tuned now
        fractions["ET"] += fractions['EU']
        fractions["XT"] += fractions['XU']

        #Eliminate them from our tables
        ptable = ptable.drop(index=['EU'], columns=['EU', 'XU'])
        fractions = fractions.drop(index=['EU', 'XU'])

        #Further make all inh tuned
        if 'inh' in mode:
            #Add new rows and columns for the new tuned inhibition
            #Get the fraction of tuned inhibitory newurons of each type
            normE = 0.0 
            normX = 0.0
            for i in range(8):

                #Create new entries in table
                ptable.loc[:, f"IT_{i}"] = 0 
                ptable.loc[f"IT_{i}", :] = 0 


                #Compute the fraction of inh neurons with this angle
                fractions[f"IT_{i}"] = fractions["I"] * fractions[f"ET_{i}"] / fractions["ET"]

                #Get the norm to scale later all I probabilities
                normE += ptable.loc["ET_0", f"ET_{i}"] * fractions[f"IT_{i}"] 
                normX += ptable.loc["ET_0", f"XT_{i}"] * fractions[f"IT_{i}"] 

            #All inh are tuned
            fractions[f"IT"] = fractions["I"] 

            #Compute how much the probability has to scale for each angle difference
            scale_E2E = [ptable.loc[f"ET_0", f"ET_{i}"] * fractions['I'] / normE for i in range(8)]
            scale_X2E = [ptable.loc[f"ET_0", f"XT_{i}"] * fractions['I'] / normX for i in range(8)]

            #Fill the new rows and columns
            for i in range(8):
                for j in range(8):
                    #Angle difference...
                    diff_angle = au.angle_dist(i, j)

                    #Presynaptic inhibition to E and I. Just take the original inh  weight and multiply by the L23 scale 
                    ptable.loc[f"ET_{i}", f"IT_{j}"] = ptable.loc[f"ET_{j}", "I"] * scale_E2E[diff_angle] 
                    ptable.loc[f"IT_{i}", f"IT_{j}"] = ptable.loc["I", "I"] * scale_E2E[diff_angle] 

                    #Postsynaptic inhibition. We do basically the same.
                    ptable.loc[f"IT_{i}", f"ET_{j}"] = ptable.loc["I", f"ET_{j}"] * scale_E2E[diff_angle] 
                    ptable.loc[f"IT_{i}", f"IT_{j}"] = ptable.loc["I", "I"] * scale_E2E[diff_angle] 
                    ptable.loc[f"IT_{i}", f"XT_{j}"] = ptable.loc["I", f"XT_{j}"] * scale_X2E[diff_angle] 

            #Now the untuned I is not necessary anymore
            ptable = ptable.drop(index=['I'], columns=['I'])

            #Put the columns in the order we like: EIX
            reordered_cols = np.hstack([np.arange(8), np.arange(16, 24), np.arange(8,16)])
            reordered_cols = ptable.columns[reordered_cols]
            ptable = ptable[reordered_cols]
        

    #presynaptic (columns) + postsynaptic (rows) names of indices that we will use to build the matrix 
    column_names = ptable.columns.values 
    row_names = ptable.index.values
    fractions = fractions[column_names]


    #The next step is to get the indices of the matrix at which each population starts
    #So for example I comes after ET and EU, meaning the first index is the sum of the amount of neurons in those populations
    #The result is two arrays array with the indices where each population starts.
    start_col = [0]
    start_row = [0] 
    prev = 0
    for col in column_names:
        start_col.append(start_col[prev] + fractions[col] * N)
        prev += 1

    #Same as above for rows
    prev = 0
    for row in row_names:
        start_row.append(start_row[prev] + fractions[row] * N)
        prev += 1

    #To int array
    start_col = np.round(start_col).astype(int)
    start_row = np.round(start_row).astype(int)

    #Get now a dictionary where each entry contains connections between the two indicated populations, 
    #which will be used to sample synaptic strength
    connections_by_group = ste.connections_per_population(units, connections)

    #Now we are ready to start sampling the matrix. Size is given by cols (which is N) and
    #rows (which depends on the number of postsynaptic neurons, it is <N)
    Q = np.zeros((start_row[-1], start_col[-1]))

    #Loop over each population in the matrix
    for postix, row in enumerate(row_names):

        #Indices where population starts and end
        r0 = start_row[postix]
        rf = start_row[postix+1]

        for preix, col in enumerate(column_names):

            #Indices where population starts and end
            c0 = start_col[preix]
            cf = start_col[preix+1]

            #Get a random matrix of the block size with the connectivity specified
            #in this population. This is just adjacency, no weights
            block = (np.random.rand(rf-r0, cf-c0) < ptable.loc[row, col]).astype(np.float64)

            #Get how many synapses this new block has, to scale them by synaptic strenght
            n_synapses = np.sum(block>0) 
        
            #syn_vols = sample_synaptic_vol(row, col, n_synapses, connections_by_group)
            #Weights depend only on cell type, which is the first letter 
            label_row = row[0]
            label_col = col[0]
            #Sample and multiply
            syn_vols = connections_by_group[f"{label_row}<{label_col}"].sample(n_synapses, replace=True)
            block[block > 0] *=  syn_vols.values

            #Negative scaling for inhibitory neurons 
            #TODO bug: the inhtuned cannot work because of the !=, while columns are now IT_X
            #flips = +1 if col != 'I' else -g
            flips = +1 if not 'I' in col else -g

            #Assign our weighted block to the matrix
            Q[r0:rf, c0:cf] = flips * J * block 

    #Change by hand the amount of inhibitory neurons!! From 5% to 3% -> a 40% reduction
    #ne, ni, nx = n_neurons
    #ni_new = round(0.6*ni)
    #Q = np.delete(Q, slice(ne+ni_new, ne+ni), axis=1) 
    #Q = np.delete(Q, slice(ne+ni_new, ne+ni), axis=0) 
    #diff = ni-ni_new
    #start_col[9:] -= diff 
    #start_row[-1] -= diff 
    #N -= diff 
    #n_neurons[1] = ni_new
    # ---

    print(N)
    print(Q.shape)
    units_sampled = sample_units(N, start_col[1:] - start_col[:-1], column_names, fractions)
    connections_sampled = sample_connections(Q)

    return units_sampled, connections_sampled, Q, n_neurons

def sample_units(N, neurons_per_pop, column_names, fractions):

    units_sampled = {'cell_type':[],'layer':[], 'pref_ori':[], 'tuning_type':[]}

    units_sampled['id'] = np.arange(N) 
    units_sampled['axon_proof'] = N*['extended'] 
    units_sampled['dendr_proof'] = N*['extended'] 

    units_sampled['osi'] = np.zeros(N)

    units_sampled['pial_dist_x'] = np.zeros(N)
    units_sampled['pial_dist_y'] = np.zeros(N)
    units_sampled['pial_dist_z'] = np.zeros(N)

    total=0
    for npop, population in zip(neurons_per_pop, column_names):
        #npop = round(N * fractions[population])

        units_sampled['layer'] += ["L4"]*npop if "X" in population else ["L23"]*npop
        units_sampled['cell_type'] += ["inh"]*npop if "I" in population else ["exc"]*npop

        if 'T' in population:
            units_sampled['pref_ori']  += [int(population[-1])]*npop 
            units_sampled['tuning_type']  += ['selective']*npop 
        else:
            units_sampled['pref_ori']  += [0]*npop 
            units_sampled['tuning_type']  += ['not_selective']*npop 

    return pd.DataFrame(data=units_sampled)


def sample_connections(Q):
    connections_sample = np.where(np.abs(Q)>0)
    connections_sample = {"pre_id":connections_sample[1], "post_id":connections_sample[0], "syn_volume":Q[connections_sample]} 
    return pd.DataFrame(connections_sample)

