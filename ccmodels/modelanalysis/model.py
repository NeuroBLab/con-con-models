import numpy as np
import scipy.integrate as scpint
from scipy.special import erf
from scipy.optimize import root
from scipy.interpolate import interp1d

import ccmodels.modelanalysis.functions_new as fun 
import ccmodels.modelanalysis.matrixsampler as msa 
import ccmodels.modelanalysis.utils as mut 
import ccmodels.dataanalysis.filters as fl


##################################################################
##### Simulate network dynamics
##################################################################

def system_euler(t, aR, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi_int_E, phi_int_I, aX, dt=0.01):
    aALL=np.empty(N_E+N_I+N_X)

    #Vector of new states of the recurrent network
    F=np.empty(np.shape(aR)[0])
    aALL[(N_E+N_I):]=aX

    doit=False

    nits = int(t/dt)
    for i in range(nits-1):
        aALL[0:(N_E+N_I)]=aR[:,i]


        #Input to the neurons
        MU_over_tau=np.matmul(QJ_ij, aALL)

        F[0:N_E] = (-aALL[0:N_E] + phi_int_E(tau_E * MU_over_tau[0:N_E])) / tau_E;
        F[N_E:]  = (-aALL[N_E:(N_E+N_I)] + phi_int_I(tau_I*MU_over_tau[N_E::])) / tau_I;

        aR[:, i+1] = aR[:,i] + dt * F

    return aR

def solve_dynamical_system(tau_E, tau_I, aX, conn, phi, dt=0.01, random_init=False):#

    T = np.arange(0, 100*tau_E, tau_I/3)

    rk45_args = [tau_E, tau_I] + conn + phi + [aX]
    QJ_ij, N_E, N_I, N_X = conn[:]

    # This function compute the dynamics of the rate model
    #Solve the model
    aR_t=np.zeros((N_E+N_I, int(np.max(T)//dt)))
    if random_init:
        aR_t[:,0] = np.random.rand(N_E+N_I) 

    aR_t = system_euler(np.max(T), aR_t, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi[0], phi[1], aX, dt)

    #Grab the solutions from scipy's IVP
    aE_t=aR_t[0:N_E, :]
    aI_t=aR_t[N_E:, :]

    #Compute observables
    aE=np.mean(aE_t[:, int(aE_t.shape[1]*0.5)::], axis=1)
    aI=np.mean(aI_t[:, int(aI_t.shape[1]*0.5)::], axis=1)

    aEstd = np.std(aE_t[:, int(aE_t.shape[1]*0.5)::], axis=1)
    aIstd = np.std(aI_t[:, int(aI_t.shape[1]*0.5)::], axis=1)

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
def do_dynamics(tau_E, tau_I, QJ, ne, ni, nx, rate_X_of_Theta, phi, dt=0.01, orionly=False, random_init=False):

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

    #Results for each stimuli
    for idx_Theta in range(ntheta):
        aX=rate_X_of_Theta[:,idx_Theta]

        Results=solve_dynamical_system(tau_E, tau_I, aX,conn, phi, dt=dt, random_init=random_init)
        ResultsALL=ResultsALL+[Results]

        aE,aI,MU_E,MU_I,aE_t,aI_t,aE_std, aI_std=Results[:]

        rate_E_of_Theta[:,idx_Theta]=aE
        rate_I_of_Theta[:,idx_Theta]=aI
        stddev_rate_E_of_Theta[:,idx_Theta]=aE_std

    return aE_t, rate_E_of_Theta, rate_I_of_Theta, stddev_rate_E_of_Theta

def make_simulation(k_ee, N, J, g, tau_E=0.02, tau_I=0.01, theta=20.0, sigma_t=10.0, V_r=10, dt=0.005, orionly=False, prepath="data", local_connectivity=True):
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

    units, connections, activity, labels = fun.statsextract(prepath=prepath, orionly=orionly)
    frac_stat, conn_stat = msa.get_fractions(units, connections, labels, local_connectivity=local_connectivity)

    scaling_prob=fun.Compute_scaling_factor_for_target_K_EE(connections, units, k_ee, N)
    #units_sampled, frac_sampled, connections_sampled = msa2.generate_functions(scaling_prob, frac_stat, conn_stat, labels, N)
    units_sampled,  connections_sampled = msa.generate_functions(scaling_prob, frac_stat, conn_stat, labels, N)
    QJ, ne, ni, nx = msa.generate_conn_matrix(units_sampled, connections_sampled, J, g)

    #Store the original preferred orientation got from the table
    tunedL23= fl.filter_neurons(units_sampled, layer='L23', tuning='tuned')
    tunedL23_ids = tunedL23['id']
    original_prefori = units_sampled.loc[units_sampled['id'].isin(tunedL23_ids), 'pref_ori']

    units_sampled.loc[units_sampled['pref_ori'].isna(), 'pref_ori'] = 0 
    units.loc[units['pref_ori'].isna(), 'pref_ori'] = 0 
    rate_xtheta = msa.sample_L4_rates(units, activity, units_sampled, orionly=orionly)

    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta, V_r, sigma_t)

    #Make simulation and do the result
    aE_t, rate_etheta, rate_itheta, stddev_rates = do_dynamics(tau_E, tau_I, QJ, ne, ni, nx, rate_xtheta, phi, dt=dt, orionly=orionly, random_init=False)


    rates_sample = np.vstack([rate_etheta, rate_itheta, rate_xtheta])
    units_sampled.loc[:, 'pref_ori'] = np.argmax(rates_sample, axis=1)

    osi = mut.compute_orientation_selectivity_index(rates_sample)

    mask_osi = osi >= 0.4
    units_sampled.loc[mask_osi, 'tuning_type'] = 'selective' 
    units_sampled.loc[~mask_osi, 'tuning_type'] = 'not_selective' 
    #units_sampled.loc[~mask_osi, 'pref_ori'] = np.random.randint(low=0, high=7, size=(~mask_osi).sum()) 

    return aE_t, rate_etheta, rate_itheta, rate_xtheta, stddev_rates, units_sampled, connections_sampled, QJ, [ne, ni, nx], tunedL23_ids, original_prefori 




def make_simulation_fixed_structure(k_ee, N, QJ, rate_xtheta, n_neurons, tau_E=0.02, tau_I=0.01, theta=20.0, sigma_t=10.0, V_r=10, dt=0.005, local_connectivity=True, orionly=False, reshuffle='no', prepath="data"):
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

    units, connections, activity, labels = fun.statsextract(prepath=prepath, orionly=orionly)
    frac_stat, conn_stat = msa.get_fractions(units, connections, labels, local_connectivity=local_connectivity)

    scaling_prob=fun.Compute_scaling_factor_for_target_K_EE(connections, units, k_ee, N)
    #units_sampled, frac_sampled, connections_sampled = msa2.generate_functions(scaling_prob, frac_stat, conn_stat, labels, N)
    units_sampled,  connections_sampled = msa.generate_functions(scaling_prob, frac_stat, conn_stat, labels, N)

    ne, ni, nx = n_neurons

    units_sampled = units_sampled.rename(columns={'pt_root_id':'id'})
    connections_sampled = connections_sampled .rename(columns={'pre_pt_root_id':'pre_id', 'post_pt_root_id':'post_id'})

    units = units.rename(columns={'pt_root_id':'id'})
    connections = connections.rename(columns={'pre_pt_root_id':'pre_id', 'post_pt_root_id':'post_id'})

    if reshuffle=='alltuned':
        ix= np.arange(0, ne+ni+nx)
        ix[0:ne] = np.random.choice(ix[0:ne], size=ne, replace=False)
        ix[ne+ni:] = np.random.choice(ix[ne+ni:], size=nx, replace=False)

        QJ_copy = QJ[:, ix]
    elif reshuffle=='L4tuned':
        ix= np.arange(0, ne+ni+nx)
        ix[ne+ni:] = np.random.choice(ix[ne+ni:], size=nx, replace=False)

        QJ_copy = QJ[:, ix]
    elif reshuffle=='L23tuned':
        ix= np.arange(0, ne+ni+nx)
        ix[0:ne] = np.random.choice(ix[0:ne], size=ne, replace=False)

        QJ_copy = QJ[:, ix]
    else:
        QJ_copy = QJ

    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta, V_r, sigma_t)

    #Make simulation and do the result
    aE_t, rate_etheta, rate_itheta, stddev_rates = do_dynamics(tau_E, tau_I, QJ_copy, ne, ni, nx, rate_xtheta, phi, dt=dt, orionly=orionly, random_init=True)

    rates = np.vstack([rate_etheta, rate_itheta, rate_xtheta])
    units_sampled.loc[:, 'pref_ori'] = np.argmax(rates, axis=1)

    osi = mut.compute_orientation_selectivity_index(rates)

    mask_osi = osi >= 0.4
    units_sampled.loc[mask_osi, 'tuning_type'] = 'selective' 
    units_sampled.loc[~mask_osi, 'tuning_type'] = 'not_selective' 

    return aE_t, rate_etheta, rate_itheta, stddev_rates, units_sampled


def make_simulation_cluster(k_ee, N, J, g, theta, sigma_t, tau_E=0.02, tau_I=0.01,  V_r=10, dt=0.005, local_connectivity=True, orionly=False, prepath="data"):
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

    units, connections, activity, labels = fun.statsextract(prepath=prepath, orionly=orionly)
    frac_stat, conn_stat = msa.get_fractions(units, connections, labels, local_connectivity=local_connectivity)

    scaling_prob=fun.Compute_scaling_factor_for_target_K_EE(connections, units, k_ee, N)
    #units_sampled, frac_sampled, connections_sampled = msa2.generate_functions(scaling_prob, frac_stat, conn_stat, labels, N)
    units_sampled,  connections_sampled = msa.generate_functions(scaling_prob, frac_stat, conn_stat, labels, N)
    QJ, ne, ni, nx = msa.generate_conn_matrix(units_sampled, connections_sampled, J, g)

    #Rename columsn to be consistent with everything else, for ease of use
    #TODO this should be done earlier in the pipeline, but it is OK
    units_sampled = units_sampled.rename(columns={'pt_root_id':'id'})
    connections_sampled = connections_sampled .rename(columns={'pre_pt_root_id':'pre_id', 'post_pt_root_id':'post_id'})

    units = units.rename(columns={'pt_root_id':'id'})
    connections = connections.rename(columns={'pre_pt_root_id':'pre_id', 'post_pt_root_id':'post_id'})

    units_sampled.loc[units_sampled['pref_ori'].isna(), 'pref_ori'] = 0 
    units.loc[units['pref_ori'].isna(), 'pref_ori'] = 0 
    rate_xtheta = msa.sample_L4_rates(units, activity, units_sampled, orionly=orionly)

    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta, V_r, sigma_t)

    #Make simulation and do the result
    aE_t, rate_etheta, rate_itheta, stddev_rates = do_dynamics(tau_E, tau_I, QJ, ne, ni, nx, rate_xtheta, phi, dt=dt, orionly=orionly, random_init=False)

    return aE_t, rate_etheta, rate_itheta, rate_xtheta, stddev_rates, units_sampled, QJ