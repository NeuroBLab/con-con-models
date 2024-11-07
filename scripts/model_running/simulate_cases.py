import numpy as np
import matplotlib.pyplot as plt

import sys
import os 
sys.path.append(os.getcwd())
import argparse

import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl


N = 8000
k = 400 

pars = [1.3, 1.0, 24.15, 9.33, 100.47, 85.42] 
mode = 'cosine'
filename = 'prueba'

units, connections, rates = loader.load_data()
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, k, N, pars[0], pars[1], hEI=pars[4], hII=pars[5],  
                                                                                                                                    theta_E=pars[2], sigma_tE=pars[3], theta_I=pars[2], sigma_tI=pars[3], cos_b=[0.175, 0.154],
                                                                                                                                    mode=mode, local_connectivity=False, orionly=True)

utl.write_synthetic_data(filename, units_sample, connections_sample, re, ri, rx, prepath="data")
np.savetxt(f'{filename}_original_pref_ori', original_prefori)


for reshuffle_mode in ['alltuned', 'L23tuned', 'L4tuned']:
    print(reshuffle_mode)
    aE_t, aI_t, re, ri, stdre, units_reshuffle, connections_reshuffle, QJ_reshuffle = md.make_simulation_fixed_structure(units_sample, connections_sample, QJ, rx, n_neurons, theta_E=pars[2], sigma_tE=pars[3], hEI=pars[4], hII=pars[5], theta_I=pars[2], sigma_tI=pars[3],
                                                                                                                                        orionly=True, reshuffle=reshuffle_mode)

    utl.write_synthetic_data(f'{filename}_{reshuffle_mode[:3]}', units_reshuffle, connections_reshuffle, re, ri, rx)