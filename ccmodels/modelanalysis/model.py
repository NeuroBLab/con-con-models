import numpy as np
import pandas as pd

import ccmodels.modelanalysis.matrixsampler as msa 
import ccmodels.modelanalysis.utils as mut 

import ccmodels.dataanalysis.processedloader as loader 
import ccmodels.dataanalysis.filters as fl

##################################################################
##### Simulate network dynamics
##################################################################

def system_euler(t, aR, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi_int_E, phi_int_I, hEI, hII, aX, dt=0.01):
    aALL=np.empty(N_E+N_I+N_X)

    #Vector of new states of the recurrent network
    F=np.empty(aR.shape[0])
    aALL[(N_E+N_I):]=aX

    nits = int(t/dt)
    for i in range(nits-1):
        aALL[0:(N_E+N_I)]=aR[:,i]


        #Input to the neurons
        MU_over_tau=np.matmul(QJ_ij, aALL)

        #F[0:N_E] = (-aALL[0:N_E] + phi_int_E(tau_E * MU_over_tau[0:N_E])) / tau_E;
        #F[N_E:]  = (-aALL[N_E:(N_E+N_I)] + phi_int_I(tau_I*MU_over_tau[N_E::])) / tau_I;
        F[0:N_E] = (-aALL[0:N_E] + phi_int_E(tau_E * (MU_over_tau[0:N_E] - hEI))) / tau_E;
        F[N_E:]  = (-aALL[N_E:(N_E+N_I)] + phi_int_I(tau_I*(MU_over_tau[N_E::] - hII))) / tau_I;

        aR[:, i+1] = aR[:,i] + dt * F

    return aR

def solve_dynamical_system(theta_sim, pref_ori, tau_E, tau_I, aX, conn, phi, hEI, hII, dt=0.01, random_init=False):#

    T = np.arange(0, 100*tau_E, tau_I/3)

    rk45_args = [tau_E, tau_I] + conn + phi + [aX]
    QJ_ij, N_E, N_I, N_X = conn[:]

    # This function compute the dynamics of the rate model
    #Solve the model
    aR_t=np.zeros((N_E+N_I, int(np.max(T)//dt)))
    if random_init:
        aR_t[:,0] = np.random.rand(N_E+N_I) 
    else:
        pass
        #aR_t[N_E:,0] = 5.0 
        #for i in range(2826):
        #    aR_t[i,0] = 1.0 + 1.0 * (theta_sim == pref_ori[i]) 
        #for i in range(2826, N_E+N_I):
        #    aR_t[i,0] = 1.0 

        #aR_t[:,0] = 1.0 + 1.0 * (theta_sim == pref_ori[:N_E+N_I]) 

    aR_t = system_euler(np.max(T), aR_t, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi[0], phi[1], hEI, hII, aX, dt)

    #Grab the solutions from scipy's IVP
    aE_t=aR_t[0:N_E, :]
    aI_t=aR_t[N_E:, :]

    #Skip thermalization
    skip = 20

    #Compute observables
    aE=np.mean(aE_t[:, skip + int(aE_t.shape[1]*0.5)::], axis=1)
    aI=np.mean(aI_t[:, skip + int(aI_t.shape[1]*0.5)::], axis=1)

    aEstd = np.std(aE_t[:, skip + int(aE_t.shape[1]*0.5)::], axis=1)
    aIstd = np.std(aI_t[:, skip + int(aI_t.shape[1]*0.5)::], axis=1)

    aALL=np.zeros(N_E+N_I+N_X)
    aALL[0:N_E]=aE
    aALL[N_E:(N_E+N_I)]=aI
    aALL[(N_E+N_I)::]=aX

    MU_over_tau = np.matmul(QJ_ij,aALL)
    MU_E = tau_E * MU_over_tau[0:N_E]
    MU_I = tau_I * MU_over_tau[N_E::]

    results=[aE, aI, MU_E, MU_I, aE_t, aI_t, aEstd, aIstd];

    return results

#def do_dynamics(Q, J_ij, m_props, rate_X_of_Theta, phi):
def do_dynamics(pref_ori, tau_E, tau_I, QJ, ne, ni, nx, rate_X_of_Theta, phi, hEI, hII, dt=0.01, orionly=False, random_init=False, return_all_series=False):

    #Initialize arguments 
    conn=[QJ, ne, ni, nx]

    #Get the angles
    if orionly:
        Theta = np.arange(0, np.pi, np.pi/8.)
    else:
        Theta = np.arange(0, 2*np.pi, np.pi/8.)

    ntheta = len(Theta)

    #Initialize rates to measure
    ResultsALL=[]
    rate_E_of_Theta=np.zeros((ne, ntheta))
    stddev_rate_E_of_Theta=np.zeros((ne, ntheta))
    rate_I_of_Theta=np.zeros((ni, ntheta))

    T = np.arange(0, 100*tau_E, tau_I/3)

    if return_all_series:
        aEt_list =np.zeros((ne, int(np.max(T)//dt), 8))
        aIt_list =np.zeros((ni, int(np.max(T)//dt), 8))

    #Results for each stimuli
    for idx_Theta in range(ntheta):
        aX=rate_X_of_Theta[:,idx_Theta]

        Results=solve_dynamical_system(idx_Theta, pref_ori, tau_E, tau_I, aX,conn, phi, hEI, hII, dt=dt, random_init=random_init)
        ResultsALL=ResultsALL+[Results]

        aE,aI,MU_E,MU_I,aE_t,aI_t,aE_std, aI_std=Results[:]

        if return_all_series:
            aEt_list[:,:,idx_Theta] = aE_t.copy()
            aIt_list[:,:,idx_Theta] = aI_t.copy()

        rate_E_of_Theta[:,idx_Theta]=aE
        rate_I_of_Theta[:,idx_Theta]=aI
        stddev_rate_E_of_Theta[:,idx_Theta]=aE_std

    if return_all_series:
        return aEt_list, aIt_list, rate_E_of_Theta, rate_I_of_Theta, stddev_rate_E_of_Theta, 
    else:
        return aE_t, rate_E_of_Theta, rate_I_of_Theta, stddev_rate_E_of_Theta

def make_simulation(units, connections, rates, k_ee, N, J, g, hEI=0.0, hII=0.0, tau_E=0.02, tau_I=0.01, 
                    theta_E=20.0, sigma_tE=10.0, theta_I=20.0, sigma_tI=10.0, V_r=10, dt=0.005, orionly=False, prepath="data", local_connectivity=True, mode='normal'):
    """
    This function makes an entire simulation for a set of parameters. It returns a sample time series for a
    single estimuli, and then the vector of rates for each one of the stimulus for E,I,X

    Parameters
    ==========
    k_ee : float
        Average number of exc to exc connections per neuron
    p_ee : float
        Probability of connection between exc neurons
    J : float
        Coupling constant
    g : float
        Balance parameter
    tau_E, tau_I : float
        Timescale of the excitatory and inhibitory neurons
    theta : float
    V_r : float
        Reset potential
    dt : float
        Timestep (default 5e-3)
    use_scipy : float
        Use scipy.initial_ivp (default, False; discouraged, as it is WAY slower than the fixed-step method)
    """

    units_sampled, connections_sampled, QJ, n_neurons = msa.sample_matrix(units, connections, k_ee, N, J, g, prepath=prepath, mode=mode)
    ne, ni, nx = n_neurons

    #Store the original preferred orientation got from the table
    tunedL23= fl.filter_neurons(units_sampled, layer='L23', tuning='tuned')
    tunedL23_ids = tunedL23['id']
    original_prefori = units_sampled.loc[units_sampled['id'].isin(tunedL23_ids), 'pref_ori']
    
    rate_xtheta = msa.sample_L4_rates(units, rates, units_sampled, mode=mode)

    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta_E, theta_I, V_r, sigma_tE, sigma_tI)

    #Make simulation and do the result
    aE_t, aI_t, rate_etheta, rate_itheta, stddev_rates = do_dynamics(units_sampled['pref_ori'].values, tau_E, tau_I, QJ, ne, ni, nx, rate_xtheta, phi, hEI, hII, 
                                                                     dt=dt, orionly=orionly, random_init=False, return_all_series=True)

    rates_sample = np.vstack([rate_etheta, rate_itheta, rate_xtheta])
    units_sampled.loc[:, 'pref_ori'] = np.argmax(rates_sample, axis=1)

    rates_L23 = np.vstack([rate_etheta, rate_itheta])
    osi = mut.compute_orientation_selectivity_index(rates_L23)
    _, cvr = mut.compute_circular_variance(rates_L23, orionly=True)
    thresori = 0.2
    mask_osi = np.where(cvr >= thresori)[0]
    print(mask_osi)
    units_sampled.loc[mask_osi, 'tuning_type'] = 'selective' 
    mask_osi = np.where(cvr < thresori)[0]
    units_sampled.loc[mask_osi, 'tuning_type'] = 'not_selective' 

    """
    osi = mut.compute_orientation_selectivity_index(rates_sample)
    
    mask_osi = osi >= 0.4
    units_sampled.loc[mask_osi, 'tuning_type'] = 'selective' 
    units_sampled.loc[~mask_osi, 'tuning_type'] = 'not_selective' 

    start = 20
    tserieslen = aE_t.shape[1]
    mitad = (tserieslen - start) // 2

    ne = n_neurons[0]

    pref_ori_1 = np.empty(ne)
    pref_ori_2 = np.empty(ne)
    for i in range(ne):
        pref_ori_1[i] = np.argmax(np.mean(aE_t[i, start:mitad+start, :], axis=0))
        pref_ori_2[i] = np.argmax(np.mean(aE_t[i, start+mitad:-1, :], axis=0))

    units_sampled.loc[:, 'tuning_type'] = 'not_selective'
    condition_selective = np.full(N, False) 
    condition_selective[:ne] = pref_ori_1 == pref_ori_2
    units_sampled.loc[condition_selective, 'tuning_type'] = 'selective'
    units_sampled.loc[ne+ni:, 'tuning_type'] = 'selective'
    """
    return aE_t, aI_t, rate_etheta, rate_itheta, rate_xtheta, stddev_rates, units_sampled, connections_sampled, QJ, [ne, ni, nx], tunedL23_ids, original_prefori 


def make_simulation_fixed_structure(units_sampled, connections_sampled, QJ, rate_xtheta, n_neurons, hEI=0., hII=0., tau_E=0.02, tau_I=0.01, 
                                    theta_E=20.0, theta_I=20, sigma_tE=10.0, sigma_tI=20.0, V_r=10, dt=0.005, orionly=False, reshuffle='no', prepath="data"):
    """
    This function makes an entire simulation for a set of parameters. It returns a sample time series for a
    single estimuli, and then the vector of rates for each one of the stimulus for E,I,X

    Parameters
    ==========
    k_ee : float
        Average number of exc to exc connections per neuron
    p_ee : float
        Probability of connection between exc neurons
    J : float
        Coupling constant
    g : float
        Balance parameter
    tau_E, tau_I : float
        Timescale of the excitatory and inhibitory neurons
    theta : float
    V_r : float
        Reset potential
    dt : float
        Timestep (default 5e-3)
    use_scipy : float
        Use scipy.initial_ivp (default, False; discouraged, as it is WAY slower than the fixed-step method)
    """

    ne, ni, nx = n_neurons

    if reshuffle=='alltuned':
        ix= np.arange(0, ne+ni+nx)
        ix[0:ne] = np.random.choice(ix[0:ne], size=ne, replace=False)
        ix[ne+ni:] = np.random.choice(ix[ne+ni:], size=nx, replace=False)

        QJ_copy = np.copy(QJ[:, ix])

        ixs = np.where(np.abs(QJ_copy) > 0)
        connections_sampled = {"pre_id" : ixs[1], "post_id": ixs[0], "syn_volume":QJ_copy[ixs[0], ixs[1]]} 
        connections_sampled = pd.DataFrame(data=connections_sampled)
    elif reshuffle=='L4tuned':
        ix= np.arange(0, ne+ni+nx)
        ix[ne+ni:] = np.random.choice(ix[ne+ni:], size=nx, replace=False)

        QJ_copy = np.copy(QJ[:, ix])

        ixs = np.where(np.abs(QJ_copy) > 0)
        connections_sampled = {"pre_id" : ixs[1], "post_id": ixs[0], "syn_volume":QJ_copy[ixs[0], ixs[1]]} 
        connections_sampled = pd.DataFrame(data=connections_sampled)
    elif reshuffle=='L23tuned':
        ix= np.arange(0, ne+ni+nx)
        ix[0:ne] = np.random.choice(ix[0:ne], size=ne, replace=False)

        QJ_copy = np.copy(QJ[:, ix])
        ixs = np.where(np.abs(QJ_copy) > 0)
        connections_sampled = {"pre_id" : ixs[1], "post_id": ixs[0], "syn_volume":QJ_copy[ixs[0], ixs[1]]} 
        connections_sampled = pd.DataFrame(data=connections_sampled)
    else:
        QJ_copy = np.copy(QJ)

    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta_E, theta_I, V_r, sigma_tE, sigma_tI)

    #Make simulation and do the result
    aE_t, aI_t, rate_etheta, rate_itheta, stddev_rates = do_dynamics(units_sampled['pref_ori'].values, tau_E, tau_I, QJ_copy, ne, ni, nx, rate_xtheta, phi, hEI, hII,  dt=dt, orionly=orionly, random_init=True, return_all_series=True)

    rates = np.vstack([rate_etheta, rate_itheta, rate_xtheta])
    units_sampled.loc[:, 'pref_ori'] = np.argmax(rates, axis=1)

    """
    osi = mut.compute_orientation_selectivity_index(rates)
    mask_osi = osi >= 0.6
    units_sampled.loc[mask_osi, 'tuning_type'] = 'selective' 
    units_sampled.loc[~mask_osi, 'tuning_type'] = 'not_selective' 
    """

    start = 20
    tserieslen = aE_t.shape[1]
    mitad = (tserieslen - start) // 2

    ne = n_neurons[0]

    pref_ori_1 = np.empty(ne)
    pref_ori_2 = np.empty(ne)
    for i in range(ne):
        pref_ori_1[i] = np.argmax(np.mean(aE_t[i, start:mitad+start, :], axis=0))
        pref_ori_2[i] = np.argmax(np.mean(aE_t[i, start+mitad:-1, :], axis=0))

    units_sampled.loc[:, 'tuning_type'] = 'not_selective'
    condition_selective = np.full(ne+ni+nx, False) 
    condition_selective[:ne] = pref_ori_1 == pref_ori_2
    units_sampled.loc[condition_selective, 'tuning_type'] = 'selective'
    units_sampled.loc[ne+ni:, 'tuning_type'] = 'selective'

    return aE_t, aI_t, rate_etheta, rate_itheta, stddev_rates, units_sampled, connections_sampled, QJ_copy


def make_simulation_cluster(units, connections, rates, k_ee, N, J, g, theta_E, theta_I, sigma_tE, sigma_tI, hEI=0., hII=0., tau_E=0.02, tau_I=0.01,  V_r=10, dt=0.005, orionly=False, prepath="data", mode='normal'):
    """
    This function makes an entire simulation for a set of parameters. It returns a sample time series for a
    single estimuli, and then the vector of rates for each one of the stimulus for E,I,X

    Parameters
    ==========
    k_ee : float
        Average number of exc to exc connections per neuron
    p_ee : float
        Probability of connection between exc neurons
    J : float
        Coupling constant
    g : float
        Balance parameter
    tau_E, tau_I : float
        Timescale of the excitatory and inhibitory neurons
    theta : float
    V_r : float
        Reset potential
    dt : float
        Timestep (default 5e-3)
    use_scipy : float
        Use scipy.initial_ivp (default, False; discouraged, as it is WAY slower than the fixed-step method)
    """

    units_sampled, connections_sampled, QJ, n_neurons = msa.sample_matrix(units, connections, k_ee, N, J, g, prepath=prepath, mode=mode)
    ne, ni, nx = n_neurons

    rate_xtheta = msa.sample_L4_rates(units, rates, units_sampled, mode=mode)

    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta_E, theta_I, V_r, sigma_tE, sigma_tI)

    #Make simulation and do the result
    aE_t, rate_etheta, rate_itheta, stddev_rates = do_dynamics(units_sampled['pref_ori'].values, tau_E, tau_I, QJ, ne, ni, nx, rate_xtheta, phi, hEI, hII, dt=dt, orionly=orionly, random_init=False)
    
    rates_sample = np.vstack([rate_etheta, rate_itheta, rate_xtheta])
    units_sampled.loc[:, 'pref_ori'] = np.argmax(rates_sample, axis=1)

    return aE_t, rate_etheta, rate_itheta, rate_xtheta, stddev_rates, units_sampled, QJ