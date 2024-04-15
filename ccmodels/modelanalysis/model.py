import numpy as np
import scipy.integrate as scpint
from scipy.special import erf
from scipy.optimize import root
from scipy.interpolate import interp1d

import ccmodels.modelanalysis.matrixsampler as msa
import ccmodels.modelanalysis.utils as mut 


##################################################################
##### Simulate network dynamics
##################################################################

def system_RK45(t, aR, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi_int_E, phi_int_I, aX):
    aALL=np.empty(N_E+N_I+N_X)
    aALL[0:(N_E+N_I)]=aR
    aALL[(N_E+N_I):]=aX

    #Input to the neurons
    MU_over_tau=np.matmul(QJ_ij, aALL)

    #Vector of new states of the recurrent network
    F=np.empty(np.shape(aR))

    F[0:N_E] = (-aALL[0:N_E] + phi_int_E(tau_E * MU_over_tau[0:N_E])) / tau_E;
    F[N_E:]  = (-aALL[N_E:(N_E+N_I)] + phi_int_I(tau_I*MU_over_tau[N_E::])) / tau_I;
    return F

def system_euler(t, aR, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi_int_E, phi_int_I, aX, dt=0.01):
    aALL=np.empty(N_E+N_I+N_X)

    #Vector of new states of the recurrent network
    F=np.empty(np.shape(aR)[0])
    aALL[(N_E+N_I):]=aX

    nits = int(t/dt)
    for i in range(nits-1):
        aALL[0:(N_E+N_I)]=aR[:,i]

        #Input to the neurons
        MU_over_tau=np.matmul(QJ_ij, aALL)

        F[0:N_E] = (-aALL[0:N_E] + phi_int_E(tau_E * MU_over_tau[0:N_E])) / tau_E;
        F[N_E:]  = (-aALL[N_E:(N_E+N_I)] + phi_int_I(tau_I*MU_over_tau[N_E::])) / tau_I;

        aR[:, i+1] = aR[:,i] + dt * F

    return aR

def solve_dynamical_system(tau_E, tau_I, aX, conn, phi, dt=0.01, use_scipy=True):#

    T = np.arange(0, 100*tau_E, tau_I/3)

    rk45_args = [tau_E, tau_I] + conn + phi + [aX]
    QJ_ij, N_E, N_I, N_X = conn[:]

    # This function compute the dynamics of the rate model
    #Solve the model
    if use_scipy:
        aR_t=np.zeros((N_E+N_I, len(T)));
        sol = scpint.solve_ivp(system_RK45, [np.min(T),np.max(T)], aR_t[:,0], method='RK45',  args=rk45_args, t_eval=T)
        aR_t=sol.y;  

        #Grab the solutions from scipy's IVP
        aE_t=aR_t[0:N_E,:]
        aI_t=aR_t[N_E::,:]
    else:
        aR_t=np.zeros((N_E+N_I, int(np.max(T)//dt)));
        aR_t = system_euler(np.max(T), aR_t, tau_E, tau_I, QJ_ij, N_E, N_I, N_X, phi[0], phi[1], aX, dt)

        #Grab the solutions from scipy's IVP
        aE_t=aR_t[0:N_E, :]
        aI_t=aR_t[N_E:, :]

    
    #Get the system's mean over the temporal standard deviation to ensure we are in a fixed point
    Convergence_aE = np.mean(np.std(aE_t[:,int(aE_t.shape[1]*2/3)::], axis=1))
    Convergence_aI = np.mean(np.std(aI_t[:,int(aI_t.shape[1]*2/3)::], axis=1))

    #Compute observables
    aE=np.mean(aE_t[:,int(aE_t.shape[1]*2/3)::],axis=1)
    aI=np.mean(aI_t[:,int(aI_t.shape[1]*2/3)::],axis=1)

    aALL=np.zeros(N_E+N_I+N_X)
    aALL[0:N_E]=aE
    aALL[N_E:(N_E+N_I)]=aI
    aALL[(N_E+N_I)::]=aX

    MU_over_tau=np.matmul(QJ_ij,aALL)
    MU_E=tau_E*MU_over_tau[0:N_E]
    MU_I=tau_I*MU_over_tau[N_E::]

    Results=[aE,aI,MU_E,MU_I,aE_t,aI_t, Convergence_aE, Convergence_aI];

    return Results

#def do_dynamics(Q, J_ij, m_props, rate_X_of_Theta, phi):
def do_dynamics(tau_E, tau_I, QJ, m_props, rate_X_of_Theta, phi, dt=0.01, use_scipy=True):

    #Initialize arguments for the solve_ivp function
    ne, ni, nx = [m_props[n] for n in ["N_E", "N_I", "N_X"]]
    conn=[QJ, ne, ni, nx]

    Theta = np.arange(0, 2*np.pi, np.pi/8.)
    ntheta = len(Theta)

    ResultsALL=[]
    rate_E_of_Theta=np.zeros((ne, ntheta))
    rate_I_of_Theta=np.zeros((ni, ntheta))

    #Results for each stimuli
    for idx_Theta in range(ntheta):
        aX=rate_X_of_Theta[:,idx_Theta]

        Results=solve_dynamical_system(tau_E, tau_I, aX,conn, phi, use_scipy=use_scipy, dt=dt)
        ResultsALL=ResultsALL+[Results]

        aE,aI,MU_E,MU_I,aE_t,aI_t, Convergence_aE, Convergence_aI=Results[:]

        rate_E_of_Theta[:,idx_Theta]=aE
        rate_I_of_Theta[:,idx_Theta]=aI

    return aE_t, rate_E_of_Theta, rate_I_of_Theta


def make_simulation(k_ee, p_ee, J, g, tau_E=0.02, tau_I=0.01, theta=20, V_r=10, dt=0.005, use_scipy=False):
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

    #Read the matrix data and construct a matrix
    data, mprops = mut.get_matrix_properties(k_ee, p_ee)
    Q, neurons_PO = msa.sample_matrix(data, mprops)

    #Sample weights of the matrix
    Jij = msa.sample_J(data, mprops, Q, J, g)

    #Get the rates for L4
    rate_xtheta = msa.get_rateX(data, mprops, neurons_PO)
    
    #Compute the response function for the used parameters
    phi = mut.tabulate_response(tau_E, tau_I, theta, V_r)

    #Make simulation and do the result
    aE_t, rate_etheta, rate_itheta = do_dynamics(tau_E, tau_I, Q*Jij, mprops, rate_xtheta, phi, dt=dt, use_scipy=use_scipy)
    return aE_t, rate_etheta, rate_itheta, rate_xtheta













