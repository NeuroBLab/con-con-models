import numpy as np
import pandas as pd

import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.statistics_extraction as ste
import ccmodels.dataanalysis.filters as fl
import ccmodels.utils.angleutils as au

is_real_data = True 


#Load the data
if is_real_data:
    v1_neurons, v1_connections, activity = loader.load_data()
else:
    v1_neurons = pd.read_csv('data/model/usampled.csv')
    v1_connections = pd.read_csv('data/model/csampled.csv')
    v1_connections["delta_ori"] = au.construct_delta_ori(v1_neurons, v1_connections, half=True)


# --- Extract connectivity statistics 

#Ask our code to generate the connection probabilities between E,I, and X populations
table = ste.estimate_conn_prob_connectomics(v1_neurons, v1_connections)

if is_real_data:
    table.to_csv("data/model/prob_connectomics_cleanaxons.csv")
else:
    table.to_csv("data/model/prob_connectomics_cleanaxons_sampled.csv")

#The same, without proofreading
table = ste.estimate_conn_prob_connectomics(v1_neurons, v1_connections, proof=[None, None])
if is_real_data:
    table.to_csv("data/model/prob_connectomics.csv")
else:
    table.to_csv("data/model/prob_connectomics_sampled.csv")

#Get the matched neurons and the inhibtory ones, together...
#In the real data, inh are not matched, so we have to put them together
if is_real_data:
    fmunits = fl.filter_neurons(v1_neurons, tuning="matched")
    inhunits = fl.filter_neurons(v1_neurons, cell_type="inh")
    fm_and_inh = pd.concat((fmunits, inhunits))

    #Use them to filter synapses that are either functionally matched OR inhibitory, thus 
    #grabbing the FM exc neurons and their inhibitory neighbours
    fmconnections = fl.synapses_by_id(v1_connections, pre_ids=fm_and_inh["id"], post_ids=fm_and_inh["id"], who="both") 

#In the model case, all the neurons are matched, so just taken them.
else:
    fm_and_inh = v1_neurons 
    fmconnections = v1_connections


#Then get the table for proofread and not proofread neurons
table = ste.estimate_conn_prob_functmatch(fm_and_inh, fmconnections)
table.index.name = "Population"
if is_real_data:
    table.to_csv("data/model/prob_funcmatch_clearaxons.csv")
else:
    table.to_csv("data/model/prob_funcmatch_clearaxons_sampled.csv")

table = ste.estimate_conn_prob_functmatch(fm_and_inh, fmconnections, proof=[None,None])
table.index.name = "Population"
if is_real_data:
    table.to_csv("data/model/prob_funcmatch.csv")
else:
    table.to_csv("data/model/prob_funcmatch_sampled.csv")

#Finally, let's get the number of neurons for each family
if is_real_data:
    table = ste.get_fraction_populations(v1_neurons)
    table.index.name = "Population"
    table.to_csv("data/model/fractions_populations.csv")
else:
    sys.exit()

# --- Get estimates for the average EE connection probability 
if is_real_data:
    avk = {}

    #Remove all autapses
    v1_connections = fl.remove_autapses(v1_connections)

    #ITerate for different layers of presynaptic neurons
    l23neurons = fl.filter_neurons(v1_neurons, layer='L23', cell_type='exc')

    #1. Similar to to ste.boostrap_prob_A2B, but with custom tunings 
    #First, get just axon clean to dendrite clean
    #dendrites = l23neurons.loc[l23neurons['dendr_proof'] != 'non', 'id']
    dendrites = l23neurons.loc[:, 'id']
    axons = l23neurons.loc[l23neurons['axon_proof'] != 'non', 'id']

    proofconnections = fl.synapses_by_id(v1_connections, pre_ids=axons, post_ids=dendrites, who='both')

    #Compute the average number of links
    potential_links = len(axons) * len(dendrites)
    p = len(proofconnections) / potential_links
    avk[f'k_clean'] = len(l23neurons) * p

    #Repeat for extended ones
    #dendrites = l23neurons.loc[l23neurons['dendr_proof'] == 'extended', 'id']
    dendrites = l23neurons.loc[:, 'id']
    axons = l23neurons.loc[l23neurons['axon_proof'] == 'extended', 'id']
    proofconnections = fl.synapses_by_id(v1_connections, pre_ids=axons, post_ids=dendrites, who='both')

    potential_links = len(axons) * len(dendrites)
    p = len(proofconnections) / potential_links
    avk[f'k_extended'] = len(l23neurons) * p

    #2. Let's just measure directly, to have a sense of lower bound.
    #Here we compute the number of inputs that a neuron receives from L23 or L4
    dendrites = v1_neurons.loc[(v1_neurons['dendr_proof']=='extended')&(v1_neurons['layer']=='L23'), 'id']
    axons = l23neurons.loc[:, 'id']
    proofconnections = v1_connections[v1_connections['pre_id'].isin(axons) & v1_connections['post_id'].isin(dendrites)] 
    avk[f'lowerbound'] = proofconnections['post_id'].value_counts().mean()

    #3. Let's use a bootstraping estimate.
    #First table contains L23 neurons as postsynaptic and any possible presynpatic neuront to them,
    #no filter. The second filters for connections between L23 neurons
    dendrites = pd.read_csv("data/in_processing/dendritas_justpost.csv")
    axons = pd.read_csv("data/in_processing/dendritas_prepost.csv")
    units = pd.read_csv('data/preprocessed/unit_table.csv')

    dendrites = dendrites[dendrites["pre_pt_root_id"] != dendrites['post_pt_root_id']]
    axons = axons[axons["pre_pt_root_id"] != axons['post_pt_root_id']]

    possible_layers = ['L1', 'L23', 'L4']

    dendrites = dendrites[dendrites['layer'].isin(possible_layers)]
    dendrites = dendrites[['pre_pt_root_id', 'post_pt_root_id']]
    axons = axons[axons['layer'].isin(possible_layers)]
    axons = axons[['pre_pt_root_id', 'post_pt_root_id']]

    l23neurons = units.loc[units['layer']=='L23', "pt_root_id"] 
    axons = axons[axons['pre_pt_root_id'].isin(l23neurons)]
    dendrites = dendrites[dendrites['post_pt_root_id'].isin(l23neurons)]

    n_bootstraps = 1000

    ids_original = dendrites[dendrites['post_pt_root_id'].isin(axons['post_pt_root_id'])]
    ids_original = ids_original['post_pt_root_id']

    n_in_connections = dendrites['post_pt_root_id'].value_counts()
    n_repeated_synapses = axons.value_counts()
    n_repeated_synapses = n_repeated_synapses.reset_index()['post_pt_root_id'].value_counts() 

    av_in_k = []
    for i in range(n_bootstraps):
        ids = ids_original.sample(frac=1, replace=True)

        kin = (n_in_connections[ids] / n_repeated_synapses[ids])
        av_in_k.append(kin.mean())

    av_in_k = np.array(av_in_k)

    avk['combined'] = np.mean(av_in_k)


    n_bootstraps = 1000

    av_in_k = []
    for i in range(n_bootstraps):
        den = dendrites.sample(frac=1, replace=True)
        axn = axons.sample(frac=1, replace=True)
        n_in_connections = den['post_pt_root_id'].value_counts()
        n_conns_per_presynap = axn.value_counts()
        av_in_k.append(n_in_connections.mean() / n_conns_per_presynap.mean())

    av_in_k = np.array(av_in_k)
    avk['upperbound'] = np.mean(av_in_k)

    pd.to_pickle(avk, 'data/model/kee.pkl')