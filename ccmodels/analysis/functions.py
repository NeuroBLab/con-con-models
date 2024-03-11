import random
import math 
import numpy as np
import scipy.integrate as scpint
from scipy.special import erf
from scipy.optimize import root
from scipy.interpolate import interp1d

##################################################################
##### Fixed parameters
##################################################################

tau_E=0.02
tau_I=0.01
theta=20
V_r=10

##################################################################
##### Single neuron transfer function
##################################################################

def tabulate_response(sigma_t=10, tau_rp=2e-3, mu_tab_max=1e4):
    """
    Compute a table with all the necessary values of the response function.
    sigma_t is the input noise in mV, determines transfer function's smoothness 
    tau_rp is the refractory period in ms
    """

    #Initialize x axis of the table
    mu_tab=np.linspace(-mu_tab_max,mu_tab_max,10000)
    mu_tab=np.concatenate(([-10**7],mu_tab))
    mu_tab=np.concatenate((mu_tab,[10**7]))

    #Compute values
    phi_tab_E,phi_tab_I=mu_tab*0,mu_tab*0;
    for idx in range(len(phi_tab_E)):
        phi_tab_E[idx]=comp_phi_tab(mu_tab[idx], tau_E,tau_rp,sigma_t)
        phi_tab_I[idx]=comp_phi_tab(mu_tab[idx], tau_I,tau_rp,sigma_t)

    #Interpolate for finer distribution
    phi_int_E = interp1d(mu_tab, phi_tab_E, kind='linear')  
    phi_int_I = interp1d(mu_tab, phi_tab_I, kind='linear')

    #phi
    return [phi_int_E,phi_int_I]



def comp_phi_tab(mu,tau,tau_rp,sigma):
    nu_prova=np.linspace(0.,1.0/tau_rp,11)
    min_u=(V_r-mu)/sigma
    max_u=(theta-mu)/sigma
    r=np.zeros(np.size(mu));
    if np.size(mu)==1:
        if(max_u<10):            
            r=1.0/(tau_rp+tau*np.sqrt(np.pi)*integrale(min_u,max_u))
        if(max_u>=10):
            r=max_u/tau/np.sqrt(np.pi)*np.exp(-max_u**2)
    if np.size(mu)>1:
        for idx in range(len(mu)):
            if(max_u[idx]<10):            
                r[idx]=1.0/(tau_rp+tau*np.sqrt(np.pi)*integrale(min_u[idx],max_u[idx]))
            if(max_u[idx]>=10):
                r[idx]=max_u[idx]/tau/np.sqrt(np.pi)*np.exp(-max_u[idx]**2)
    return r

def f(x):
    param=-5.5
    if (x>=param):
        return np.exp(x**2)*(1+erf(x))
    if (x<param):
        return -1/np.sqrt(np.pi)/x*(1.0-1.0/2.0*pow(x,-2.0)+3.0/4.0*pow(x,-4.0))

def integrale(minimo,massimo):
    if massimo<11:
        adelleh=scpint.quad(lambda u: f(u),minimo,massimo)
    if massimo>=11:
        adelleh=[1./massimo*np.exp(massimo**2)]
    return adelleh[0]

##################################################################
##### Simulate network dynamics
##################################################################

def system_RK45(t, aR, QJ_ij, N_E, N_I, N_X, phi_int_E, phi_int_I, aX):
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

def solve_dynamical_system(aX,conn,phi,T=np.arange(0,100*tau_E,tau_I/3)):#T=np.arange(0,100*tau_E,tau_I/3)):#

    #[QJ_ij, N_E, N_I, N_X] = conn[:]
    #phi_int_E, phi_int_I   = phi[:]

    rk45_args = conn + phi + [aX]
    QJ_ij, N_E, N_I, N_X = conn[:]

    # This function compute the dynamics of the rate model

    #Solve the model
    aR_t=np.zeros((N_E+N_I,len(T)));
    sol = scpint.solve_ivp(system_RK45, [np.min(T),np.max(T)], aR_t[:,0], method='RK45',  args=rk45_args, t_eval=T)

    #Grab the solutions from scipy's IVP
    aR_t=sol.y;  
    aE_t=aR_t[0:N_E,:]
    aI_t=aR_t[N_E::,:]
    
    #Get the system's mean over the temporal standard deviation to ensure we are in a fixed point
    Convergence_aE=np.mean(np.std(aE_t[:,int(np.shape(aE_t)[1]*2/3)::],axis=1))
    Convergence_aI=np.mean(np.std(aI_t[:,int(np.shape(aI_t)[1]*2/3)::],axis=1))

    #Compute observables
    aE=np.mean(aE_t[:,int(np.shape(aE_t)[1]*2/3)::],axis=1)
    aI=np.mean(aI_t[:,int(np.shape(aI_t)[1]*2/3)::],axis=1)
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
def do_dynamics(QJ, m_props, rate_X_of_Theta, phi):

    #Initialize arguments for the solve_ivp function
    ne, ni, nx = [m_props[n] for n in ["N_E", "N_I", "N_X"]]
    #conn=[Q*J_ij, ne, ni, nx]
    conn=[QJ, ne, ni, nx]

    Theta = np.arange(0, 2*np.pi, np.pi/8.)
    ntheta = len(Theta)

    ResultsALL=[]
    rate_E_of_Theta=np.zeros((ne, ntheta))
    rate_I_of_Theta=np.zeros((ni, ntheta))
    for idx_Theta in range(ntheta):
        aX=rate_X_of_Theta[:,idx_Theta]
        Results=solve_dynamical_system(aX,conn, phi)
        ResultsALL=ResultsALL+[Results]
        aE,aI,MU_E,MU_I,aE_t,aI_t, Convergence_aE, Convergence_aI=Results[:]
        rate_E_of_Theta[:,idx_Theta]=aE
        rate_I_of_Theta[:,idx_Theta]=aI

    return aE_t, rate_E_of_Theta, rate_I_of_Theta


def make_simulation(k_ee, p_ee, J, g):
    data, mprops = get_matrix_properties(k_ee, p_ee)
    Q, neurons_PO = sample_matrix(data, mprops)
    #print(Q.sum(), " ", Q.mean())
    #print(Q)
    #print(np.nanmean(neurons_PO))
    Jij = sample_J(data, mprops, Q, J, g)
    rate_xtheta = get_rateX(data, mprops, neurons_PO)
    #print(rate_xtheta)
    #print(Jij.sum(), " ", Jij.mean())
    #print(Jij)
    phi = tabulate_response()
    #print(phi)
    #aE_t, rate_etheta, rate_itheta = do_dynamics(Q, Jij, mprops, rate_xtheta, phi)
    aE_t, rate_etheta, rate_itheta = do_dynamics(Q*Jij, mprops, rate_xtheta, phi)
    return aE_t, rate_etheta, rate_itheta, rate_xtheta


def opendata(i, j, returnact=False, path="."):
    J_values = np.logspace(-1, 1, 100)
    g_values = np.linspace(1.5, 7.5, 10)
    J = J_values[i]
    g = g_values[j]

    if returnact:
        return [np.load(f"{path}/aEt_{J:.2f}_{g:.2f}.npy"), np.load(f"{path}/rE_{J:.2f}_{g:.2f}.npy"),
                np.load(f"{path}/rI_{J:.2f}_{g:.2f}.npy"), np.load(f"{path}/rX_{J:.2f}_{g:.2f}.npy")]
    else:
        return [np.load(f"{path}/rE_{J:.2f}_{g:.2f}.npy"), np.load(f"{path}/rI_{J:.2f}_{g:.2f}.npy"), np.load(f"{path}/rX_{J:.2f}_{g:.2f}.npy")]

##################################################################
##### Other auxiliary functions 
##################################################################

def compute_orientation_selectivity_index(A, Theta):# shape of A: (N X 16)
    
    Max=np.max(A,axis=1)
    idx_Max=np.argmax(A,axis=1)
    idx_Min=idx_Max+4
    idx_Min[idx_Min>=len(Theta)]=idx_Max[idx_Min>=len(Theta)]-4
    Min=A[np.arange(A.shape[0]), idx_Min]

    OSI=(Max-Min)/(np.abs(Max)+np.abs(Min))
    return OSI



def get_matrix_properties(k_EE, p_EE = 0.1, assign_chi=False, path_2_propdata='data/sample_data.pkl'):
    """
    Read the data stored in path_2_propdata, which contains all the statistical data (fractions of neurons and connections
    of each type), and from the two free parameters reconstruct all the missing parameters (number of neurons, average
    connectivity, probs of connection) and returns first the original data and then the new properties as 
    dictionaries.
    """

    data = np.load(path_2_propdata, allow_pickle=True)

    matrix_props = {}

    #Number of neurons given the free parameters 
    matrix_props["N_E"] = int(k_EE/p_EE)
    matrix_props["N_I"] = int(data["eta_I"] * matrix_props["N_E"]) #eta is the fraction N_A / N_E, for A={I,X} 
    matrix_props["N_X"] = int(data["eta_X"] * matrix_props["N_E"]) 

    #Average connectivity to E neurons
    matrix_props["k_E_TOT"] = k_EE / data["gamma_EE"]
    matrix_props["k_EE"] = k_EE
    matrix_props["k_EI"] = data["gamma_EI"] * matrix_props["k_E_TOT"]
    matrix_props["k_EX"] = data["gamma_EX"] * matrix_props["k_E_TOT"]

    #Get the average degree of inhibitory neurons 
    #(encoded in the fraction chi = ki/ke)
    if assign_chi:
        chi=1.0
    else:
        chi = data["chi"]

    matrix_props["k_I_TOT"] = chi * matrix_props["k_E_TOT"]

    #Fractions of E,I,X inputs to I neurons
    matrix_props["k_IE"] = data["gamma_IE"] * matrix_props["k_I_TOT"]
    matrix_props["k_II"] = data["gamma_II"] * matrix_props["k_I_TOT"]
    matrix_props["k_IX"] = data["gamma_IX"] * matrix_props["k_I_TOT"]


    #Connection probabilities for the different populations
    matrix_props["p_EE"] = p_EE
    matrix_props["p_EI"] = data["gamma_EI"] * matrix_props["k_E_TOT"] / matrix_props["N_I"]
    matrix_props["p_EX"] = data["gamma_EX"] * matrix_props["k_E_TOT"] / matrix_props["N_X"]

    matrix_props["p_IE"] = data["gamma_IE"] * matrix_props["k_I_TOT"] / matrix_props["N_E"]
    matrix_props["p_II"] = data["gamma_II"] * matrix_props["k_I_TOT"] / matrix_props["N_I"]
    matrix_props["p_IX"] = data["gamma_IX"] * matrix_props["k_I_TOT"] / matrix_props["N_X"]

    # Compute quantities for EX Tuned Untuned network
    #First, number of tuned/untuned
    matrix_props["NT_E"] = int(data["etaT_E"] * matrix_props["N_E"])
    matrix_props["NT_X"] = int(data["etaT_X"] * matrix_props["N_X"])
    matrix_props["NU_E"] = matrix_props["N_E"] - matrix_props["NT_E"]
    matrix_props["NU_X"] = matrix_props["N_X"] - matrix_props["NT_X"]

    #Average connectivity of Tuned neurons
    matrix_props["kTT_EE"] = data["gammaTT_EE"] * matrix_props['k_EE']
    matrix_props["kTT_EX"] = data["gammaTT_EX"] * matrix_props['k_EX']

    #Connection probabilities
    matrix_props["pTT_EE"] = data["gammaTT_EE"] * matrix_props["k_EE"] / matrix_props["NT_E"]
    matrix_props["pTU_EE"] = data["gammaTU_EE"] * matrix_props["k_EE"] / matrix_props["NU_E"]
    matrix_props["pUT_EE"] = data["gammaUT_EE"] * matrix_props["k_EE"] / matrix_props["NT_E"]
    matrix_props["pUU_EE"] = data["gammaUU_EE"] * matrix_props["k_EE"] / matrix_props["NU_E"]

    matrix_props["pTT_EX"] = data["gammaTT_EX"] * matrix_props["k_EX"] / matrix_props["NT_X"]
    matrix_props["pTU_EX"] = data["gammaTU_EX"] * matrix_props["k_EX"] / matrix_props["NU_X"]
    matrix_props["pUT_EX"] = data["gammaUT_EX"] * matrix_props["k_EX"] / matrix_props["NT_X"]
    matrix_props["pUU_EX"] = data["gammaUU_EX"] * matrix_props["k_EX"] / matrix_props["NU_X"]

    #Return results
    return data, matrix_props

def gammaDTheta_EE(data, x):
    if np.isscalar(x):  # Check if x is a single value
        closest_index = np.argmin(np.abs(data["delta_PO"] - x))
        closest_value = data["delta_PO"][closest_index]
        proximity_check = abs(closest_value - x) <= 0.01
        result = data["gammaDTheta_EE"][closest_index] if proximity_check else np.nan
    else:
        closest_indices = np.argmin(np.abs(data["delta_PO"][:, None] - x), axis=0)
        closest_values = data["delta_PO"][closest_indices]
        proximity_check = np.abs(closest_values - x) <= 0.01
        result = np.where(proximity_check, data["gammaDTheta_EE"][closest_indices], np.nan)
    return result

def gammaDTheta_EX(data, x):
    if np.isscalar(x):  # Check if x is a single value
        closest_index = np.argmin(np.abs(data["delta_PO"] - x))
        closest_value = data["delta_PO"][closest_index]
        proximity_check = abs(closest_value - x) <= 0.01
        result = data["gammaDTheta_EX"][closest_index] if proximity_check else np.nan
    else:
        closest_indices = np.argmin(np.abs(data["delta_PO"][:, None] - x), axis=0)
        closest_values = data["delta_PO"][closest_indices]
        proximity_check = np.abs(closest_values - x) <= 0.01
        result = np.where(proximity_check, data["gammaDTheta_EX"][closest_indices], np.nan)
    return result

def sample_matrix(data, m_props):
    """
    Given matrix properties, get a sample of the matrix
    """

    #Handy notation
    N = [m_props[n] for n in ["N_E", "N_I", "N_X"]]
    ne, ni, nx = N 

    #Connection probabilities in each block
    P=np.empty((2, 3));
    P[0,:] = [m_props[pAB] for pAB in ["p_EE", "p_EI" , "p_EX"]]
    P[1,:] = [m_props[pAB] for pAB in ["p_IE", "p_II" , "p_IX"]]

    Q=np.nan*np.ones((ne + ni, ne+ni+nx))
    idx=[np.arange(0,ne,1), np.arange(ne, ne+ni, 1), np.arange(ne+ni, ne+ni+nx,1)]

    # Fill matrix only based on the neurons types
    for A in range(2):
        for B in range(3):
            idx_Post = idx[A]
            idx_Pre  = idx[B]
            Q[idx_Post[:, None], idx_Pre] = np.random.binomial(1, P[A,B], (N[A],N[B]))
    
    #Tuned-untuned neurons
    P=np.zeros((2, 2));
    P[0,:] = [m_props[pAB] for pAB in ["pTT_EE", "pTU_EE"]]
    P[1,:] = [m_props[pAB] for pAB in ["pUT_EE", "pUU_EE"]]

    N=[m_props[n] for n in ["NT_E", "NU_E"]]
    nte, nue = N
    idx=[np.arange(0, nte, 1), np.arange(nte, ne, 1)]

    for A in range(2):
        for B in range(2):
            idx_Post = idx[A]
            idx_Pre  = idx[B]
            Q[idx_Post[:, None],idx_Pre] = np.random.binomial(1, P[A,B], (N[A], N[B]))


    P=np.zeros((2, 2));
    P[0,:] = [m_props[pAB] for pAB in ["pTT_EX", "pTU_EX"]]
    P[1,:] = [m_props[pAB] for pAB in ["pUT_EX", "pUU_EX"]]


    #Tuned-untuned from layer 4
    N_1=[m_props[n] for n in ["NT_E", "NU_E"]]
    N_2=[m_props[n] for n in ["NT_X", "NU_X"]]

    nte, nue = N_1
    ntx, nux = N_2

    idx_1=[np.arange(0, nte, 1),np.arange(nte, ne, 1)]
    idx_2=[np.arange(ne+ni, ne+ni+ntx,1), np.arange(ne+ni+ntx, ne+ni+nx, 1)]

    for A in range(2):
        for B in range(2):
            idx_Post=idx_1[A]
            idx_Pre=idx_2[B]
            Q[idx_Post[:, None],idx_Pre]=np.random.binomial(1, P[A,B], (N_1[A],N_2[B]))
        

    #Modify connectivity based on tuning properties...
    Theta = data["Theta"] 
    ntheta = len(Theta)

    #Assign preferred orientation to each neuron
    Assigned_PO=np.nan*np.ones(ne+ni+nx)
    NhatTheta_E = int(nte/ntheta)
    NhatTheta_X = int(ntx/ntheta)
    for i in range(ntheta):
        start_index = i * NhatTheta_E
        end_index = (i + 1) * NhatTheta_E
        Assigned_PO[start_index:end_index] = Theta[i]
        
        start_index = ne+ni + i * NhatTheta_X
        end_index = ne+ni + (i + 1) * NhatTheta_X
        Assigned_PO[start_index:end_index] = Theta[i]

    #Perform the modification
    for i in range(ntheta):
        for j in range(ntheta):

            #Get the difference in PO between this two and normalize
            dPO=np.round(Theta[i]-Theta[j], 3)

            if dPO>=np.pi:
                dPO -= 2 * np.pi      
            if dPO<-np.pi*(1.001):
                dPO += 2 * np.pi      
                
            #Change the matrix
            PhatDTheta_E = gammaDTheta_EE(data, dPO)* m_props["kTT_EE"]/ NhatTheta_E
            idx_Post=np.arange(i * NhatTheta_E,(i + 1) * NhatTheta_E,1)
            idx_Pre=np.arange(j * NhatTheta_E,(j + 1) * NhatTheta_E,1)
            Q[idx_Post[:, None],idx_Pre]=np.random.binomial(1,PhatDTheta_E, (NhatTheta_E,NhatTheta_E))
            
            PhatDTheta_X=gammaDTheta_EX(data, dPO) * m_props["kTT_EX"]/NhatTheta_X
            idx_Post=np.arange(i * NhatTheta_E,(i + 1) * NhatTheta_E,1)
            idx_Pre=np.arange(ne+ni+j * NhatTheta_X,ne+ni+(j + 1) * NhatTheta_X,1)
            Q[idx_Post[:, None],idx_Pre]=np.random.binomial(1,PhatDTheta_X, (NhatTheta_E,NhatTheta_X))

    

    return Q, Assigned_PO

def get_rateX(data, m_props, Assigned_PO):
    nx = m_props["N_X"]
    Theta = data["Theta"]
    ntheta = len(Theta)
    NhatTheta_X = int(m_props["NT_X"]/ntheta)

    rate_X_of_Theta=np.nan*np.ones((nx, ntheta))
    # firing rates of tuned neurons
    for idx_Theta in range(ntheta):

        idx=np.where(Assigned_PO==Theta[idx_Theta])[0]
        idx=np.arange(NhatTheta_X*idx_Theta,NhatTheta_X*(idx_Theta+1),1)
        
        idx_sample=np.random.randint(0, np.shape(data["sampled_ratesT_X"])[0] , size=len(idx))

        rate_X_of_Theta[idx,:]=np.roll(data["sampled_ratesT_X"][idx_sample,:],idx_Theta,axis=1)
    # firing rates of untuned neurons
    idx_sample=np.random.randint(0, np.shape(data["sampled_ratesU_X"])[0] , size=len(np.arange(idx[-1]+1, nx,1)))
    rate_X_of_Theta[(idx[-1]+1)::,:]=data["sampled_ratesU_X"][idx_sample,:]

    return rate_X_of_Theta



def sample_J(data, m_props, Q, J, g):
    #J=3.;g=1.5;
    ne, ni, nx = [m_props[n] for n in ["N_E", "N_I", "N_X"]]
    idx_E=np.arange(0,ne, 1)
    idx_I=np.arange(ne, ne+ni, 1)
    idx_X=np.arange(ne + ni, ne+ni+nx,1)

    J_ij=np.nan*np.ones(np.shape(Q))

    J_ij[np.ix_(idx_E,idx_E)] = J  * np.random.choice(data["sampled_J_EE"], size=np.shape(Q[np.ix_(idx_E,idx_E)]))
    J_ij[np.ix_(idx_E,idx_I)] = -J*g*np.random.choice(data["sampled_J_EI"], size=np.shape(Q[np.ix_(idx_E,idx_I)]))
    J_ij[np.ix_(idx_E,idx_X)] = J  * np.random.choice(data["sampled_J_EX"], size=np.shape(Q[np.ix_(idx_E,idx_X)]))

    J_ij[np.ix_(idx_I,idx_E)] = J  * np.random.choice(data["sampled_J_IE"], size=np.shape(Q[np.ix_(idx_I,idx_E)]))
    J_ij[np.ix_(idx_I,idx_I)] = -J*g*np.random.choice(data["sampled_J_II"], size=np.shape(Q[np.ix_(idx_I,idx_I)]))
    J_ij[np.ix_(idx_I,idx_X)] = J  * np.random.choice(data["sampled_J_IX"], size=np.shape(Q[np.ix_(idx_I,idx_X)]))

    return J_ij

