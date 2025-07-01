import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

#Von mises function for direction and orientation...
def von_mises_dir(x, k, m, a1, a2, b):
    return a1*np.exp(k*np.cos(x-m)) + a2*np.exp(k*np.cos(x-m+np.pi)) + b

def von_mises_ori(x, k, m, a, b):
    return a*np.exp(k*np.cos(2*(x-m))) + b

#Function to curve_fit an average tuning curve
def fit_ori(thetas_ori, ydata):

    #Initial guess, using some info from the data
    k = 7
    max_pos = np.argmax(ydata)
    m = thetas_ori[max_pos] 
    a = ydata[max_pos] 
    b = ydata[np.argmin(ydata)] 

    p0 = [k, m, a, b]

    #Try helps us in case curve_fit fails without cutting the program
    try:
        #Fit the orientation function with the correct bounds
        popt, _ = curve_fit(von_mises_ori, thetas_ori, ydata, p0=p0, bounds=(0, np.inf), maxfev=2000)

        #Compute the R^2 of the model from its definition and return it
        residuals = np.sum((ydata - von_mises_ori(thetas_ori, *popt))**2)
        sumtotal  = np.sum((ydata - ydata.mean())**2) 

        r2 = 1 - residuals / sumtotal

        return popt, r2
    except:
        #If it fails, return R^2 = 0 and no parameters
        return np.zeros(len(p0)), 0.

#Same as above but for direction
def fit_dir(thetas_dir, ydata):

    #Initial guess
    k = 7
    max_pos = np.argmax(ydata)
    m = thetas_dir[max_pos] 
    a = ydata[max_pos] 
    b = ydata[np.argmin(ydata)] 

    #Notice secondary peak is assumed to be 1/10 of the largest one
    p0 = [k, m, a, a/10, b]

    #Same as in the function above
    try:
        popt, _ = curve_fit(von_mises_dir, thetas_dir, ydata, p0=p0, bounds=(0, np.inf), maxfev=2000)

        residuals = np.sum((ydata - von_mises_dir(thetas_dir, *popt))**2)
        sumtotal  = np.sum((ydata - ydata.mean())**2) 

        r2 = 1 - residuals / sumtotal

        return popt, r2
    except:
        return np.zeros(len(p0)), 0.

#Test if 
def test_ori(n_neurons, thetas_ori, response_stacked_trials, oris_stacked_trials, params_neuron):
    pvals    = np.empty(n_neurons)
    pref_ori = np.empty(n_neurons, dtype=int)

    #Count how many times each orientation appears. There are few frames of difference for each one...
    #We will use the minimum one in order to be able to always substract and test 
    oris, counts = np.unique(oris_stacked_trials, return_counts=True)
    min_len = np.min(counts)

    #Precompute the differences between responses for each angle and its angle + pi/2
    diffs_precomputed = np.empty((n_neurons, min_len, 8))
    for ori in range(8):
        mid_ori = (ori + 4) % 8
        mask_max  = oris_stacked_trials == ori 
        mask_mid  = oris_stacked_trials == mid_ori 

        #These two DO have different lengths because there is a different # of ori and mid_ori
        response_max = response_stacked_trials[:, mask_max]
        response_mid = response_stacked_trials[:, mask_mid]

        #So clip them down to the minimum length when substracting
        diffs_precomputed[:, :, ori] = response_max[:, :min_len] - response_mid[:, :min_len] 

    #Check for every neuron
    for i in range(n_neurons):
        #Estimate the preferred orientation from the fit
        fit = von_mises_ori(thetas_ori, *params_neuron[i, :])
        pref_ori[i]   = np.argmax(fit) 

        #Check the difference between pref_ori and pref_ori + pi/2
        stat,pvals[i] = wilcoxon(diffs_precomputed[i, :, pref_ori[i]])

    #Return
    return pref_ori, pvals

#Same as the function above, but for directions
def test_dir(n_neurons, thetas_dir, response_stacked_trials, dirs_stacked_trials, params_neuron):
    #Now we have to check differences for two, one for + pi/2 and another for +pi
    pvals_mid  = np.empty(n_neurons)
    pvals_anti = np.empty(n_neurons)
    pref_dir   = np.empty(n_neurons, dtype=int)

    #Minimum number of trials...
    dirs, counts = np.unique(dirs_stacked_trials, return_counts=True)
    min_len = np.min(counts)

    #Precompute differences 
    diffs_mid_precomputed  = np.empty((n_neurons, min_len, 16))
    diffs_anti_precomputed = np.empty((n_neurons, min_len, 16))
    for dir in range(16):
        mid_dir =  (dir + 4) % 16
        anti_dir = (dir + 8) % 16
        mask_max  = dirs_stacked_trials == dir 
        mask_mid  = dirs_stacked_trials == mid_dir 
        mask_anti = dirs_stacked_trials == anti_dir

        response_max  = response_stacked_trials[:n_neurons, mask_max]
        response_mid  = response_stacked_trials[:n_neurons, mask_mid]
        response_anti = response_stacked_trials[:n_neurons, mask_anti]

        diffs_mid_precomputed[:, :, dir]  = response_max[:, :min_len] - response_mid[:, :min_len] 
        diffs_anti_precomputed[:, :, dir] = response_max[:, :min_len] - response_anti[:, :min_len] 

    #Get preferred direction and test differences!
    for i in range(n_neurons):
        fit = von_mises_dir(thetas_dir, *params_neuron[i, :])
        pref_dir[i]        = np.argmax(fit) 
        stat,pvals_mid[i]  = wilcoxon(diffs_mid_precomputed[i, :,  pref_dir[i]])
        stat,pvals_anti[i] = wilcoxon(diffs_anti_precomputed[i, :, pref_dir[i]])

    return pref_dir, pvals_mid, pvals_anti