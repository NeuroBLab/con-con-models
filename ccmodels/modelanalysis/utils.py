import numpy as np
import scipy.integrate as scpint
from scipy.special import erf
from scipy.interpolate import interp1d

##################################################################
##### Single neuron transfer function
##################################################################

def tabulate_response(tau_E, tau_I, theta, V_r, sigma_t = 10, tau_rp = 2e-3, xf=1e4, xinf=1e7, nsteps=10000):
    """
    Compute a table with all the necessary values of the response function.
    sigma_t is the input noise in mV, determines transfer function's smoothness 
    tau_rp is the refractory period in ms.

    Parameters
    ==========
    tau_E, tau_I : float
        Timescale of the E and I populations, respectively
    theta : float
        Parameter theta
    V_r : float
        Resting potential
    sigma_t : float 
        Amount of input noise in mV, determines transfer function's smoothness
    tau_rp : float
        Refractory period in ms
    xf : float
        The table will be constructed in the interval [-xf, xf]. Default is 1e4
    xinf : float
        A value to represent the infinite. Should be way larger than xf (default 1e7)
    nsteps : int
        The number of steps to divide the interval [-xf, xf] in
    """

    #Initialize x axis of the table
    mu_tab=np.linspace(-xf, xf, nsteps)
    mu_tab=np.concatenate(([-xinf], mu_tab))
    mu_tab=np.concatenate((mu_tab, [xinf]))

    #Compute values
    #phi_tab_E, phi_tab_I = mu_tab*0,mu_tab*0;
    phi_tab_E = np.zeros(len(mu_tab))
    phi_tab_I = np.zeros(len(mu_tab))

    #Compute the function at each one of the points
    for idx in range(mu_tab.size):
        phi_tab_E[idx] = comp_phi_tab(mu_tab[idx], tau_E, tau_rp, sigma_t, V_r, theta)
        phi_tab_I[idx] = comp_phi_tab(mu_tab[idx], tau_I, tau_rp, sigma_t, V_r, theta)

    #Interpolate for finer distribution
    phi_int_E = interp1d(mu_tab, phi_tab_E, kind='linear')  
    phi_int_I = interp1d(mu_tab, phi_tab_I, kind='linear')

    #Return 
    return [phi_int_E, phi_int_I]



def comp_phi_tab(mu, tau, tau_rp, sigma, V_r, theta, u_thres=10):
    """
    For each value of the given variables, get the value of the integral

    Parameters
    ==========

    mu : float
    tau : float
    tau_rp : float
        Refractory period
    sigma : float
    V_r : float
        Resting potential
    theta : float
    u_thres : float
        threshold to start making an approximation for the integral
        
    """

    min_u=(V_r-mu)/sigma
    max_u=(theta-mu)/sigma

    if(max_u < u_thres):            
        return 1.0/(tau_rp+tau*np.sqrt(np.pi)*compute_integral(min_u,max_u))
    else:
        return max_u/tau/np.sqrt(np.pi)*np.exp(-max_u**2)
    return r


def integrand(x, thres = -5.5):
    """
    The function to integrate, which is computed approximately.

    Parameters
    ==========
    x : float
        integration variable
    thres : float
        If x > thres (default -5.5), an approximation to the integrand is returned instead.
    """

    if (x>=thres):
        return np.exp(x*x)*(1 + erf(x))
    else:
        return (0.5*x**(-2) - 0.75*x**(-4) - 1) / (np.sqrt(np.pi) * x)  


def compute_integral(min, max, thres=11.):
    """
    Compute the integral under an aproximation.


    Parameters
    ==========
    min, max : float
        Limits of the integral
    thres : float
        If max > thres (default 11), the integral is approximated by exp(max^2)/max
    """

    if max < thres:
        integral = scpint.quad(integrand, min, max)
        return integral[0]
    else: 
        return np.exp(max*max)/max



##################################################################
##### Loading data from disk 
##################################################################


def opendata(i, j, returnact=False, path="."):
    """
    Open stored data from the disk for a value (J, g) using the corresponding indices.
    Activity is returned only if returnact is True
    """

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

def compute_orientation_selectivity_index(rates):
    """
    Compute the OSI of the selected rates.

    Parameters
    ===========
    rates : numpy array
        The classical Nxnangles matrix with the rates 
    """

    #How many angles are we considering?
    n, nangles = rates.shape

    idx_Max=np.argmax(rates, axis=1)

    #Check OSI at +pi/2
    idx_Min = (idx_Max + 4) % nangles


    OSI = np.empty(n)
    for k,i in enumerate(range(n)):
        Max = rates[i, idx_Max[k]]
        Min = rates[i, idx_Min[k]]
        OSI[k] = (Max-Min)/(Max+Min)

    return OSI


def compute_circular_variance(rates, orionly=False):
    """
    Compute the OSI of the selected rates.

    Parameters
    ===========
    rates : numpy array
        The classical Nxnangles matrix with the rates 
    """

    if orionly:
        angles = np.arange(0, 2*np.pi, np.pi/4)
    else:
        angles = np.arange(0, 2*np.pi, np.pi/8)


    if len(rates.shape) > 1:
        norma = rates.sum(axis=1)
        num_ori = rates * np.exp(2.0j * angles)
        num_dir = rates * np.exp(1.0j * angles)
        r_ori = np.abs(num_ori.sum(axis=1)/norma)
        r_dir = np.abs(num_dir.sum(axis=1)/norma)
    else:
        norma = rates.sum()
        num_ori = rates * np.exp(2.0j * angles)
        num_dir = rates * np.exp(1.0j * angles)
        r_ori = np.abs(num_ori.sum()/norma)
        r_dir = np.abs(num_dir.sum()/norma)

    return r_ori, r_dir