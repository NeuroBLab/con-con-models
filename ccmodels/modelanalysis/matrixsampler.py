import numpy as np
import math

import functions as fun

import random
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, ks_2samp

# load files
def get_fractions(unit_table, connections_table, Labels):

    Frac_tables_data=fun.measure_fractions_of_neurons(unit_table,Labels)
    Conn_stat_measured_on_data=fun.measure_connection_stats(connections_table, unit_table,Labels)
    return Frac_tables_data, Conn_stat_measured_on_data 

#TODO this has to be called from the parent script between the previous two functions
#N=1000, target_K_EE=200
#scaling_prob=fun.Compute_scaling_factor_for_target_K_EE(connections_table, unit_table,target_K_EE,N)

def generate_functions(scaling_prob, Frac_tables_data, Conn_stat_measured_on_data, Labels, N=1000):

    # sample and measure stats on sampled quantities
    neurons_sampled=fun.sample_neurons_with_tuning(Frac_tables_data,N,Labels)
    Frac_tables_sampled=fun.measure_fractions_of_neurons(neurons_sampled,Labels)
    sampled_connections=fun.sample_connections(Conn_stat_measured_on_data,neurons_sampled,scaling_prob,Labels)

    return neurons_sampled,Frac_tables_sampled, sampled_connections

    #Conn_stat_measured_on_sampled=fun.measure_connection_stats(sampled_connections,neurons_sampled,Labels)

def generate_conn_matrix(neurons_sampled, sampled_connections, J, g):
    # Initialize QJ array with zeros
    QJ = np.zeros((len(neurons_sampled), len(neurons_sampled)))

    # Get the necessary data from sampled_connections
    pre_pt_root_ids = sampled_connections['pre_pt_root_id']
    post_pt_root_ids = sampled_connections['post_pt_root_id']
    syn_volumes = sampled_connections['syn_volume']

    # Assign synapse volumes to QJ array based on pre and post synaptic root IDs and scale by factor J
    QJ[post_pt_root_ids, pre_pt_root_ids] = J*syn_volumes

    # scale inhibitory connections and make them negative
    num_L23_neurons_E = len(neurons_sampled[(neurons_sampled['layer'] == 'L23')&(neurons_sampled['cell_type'] =='exc')])
    num_L23_neurons_I = len(neurons_sampled[(neurons_sampled['layer'] == 'L23')&(neurons_sampled['cell_type'] =='inh')])
    QJ[:, num_L23_neurons_E:(num_L23_neurons_E+num_L23_neurons_I)]=-g*QJ[:, num_L23_neurons_E:(num_L23_neurons_E+num_L23_neurons_I)]

    # Remove post synaptic neurons in L4
    num_L23_neurons = len(neurons_sampled[neurons_sampled['layer'] == 'L23'])
    QJ = QJ[:num_L23_neurons, :]
    Q = QJ.copy()
    Q[QJ!=0]=1

    return Q, QJ