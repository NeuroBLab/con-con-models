import numpy as np

##################################################################
##### Matrix Sampling 
##################################################################

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