import numpy as np
import pandas as pd 
from microns_phase3 import nda
from scipy.stats import wilcoxon

def cell_area_identifiers(brain_area):
    '''brain_area: str, name of brain area for which cells are of interes, can be one of  'AL', 'LM', 'RL', 'V1' '''
    areas = nda.AreaMembership().fetch(format = 'frame').reset_index()
    areas_clean = areas[areas['brain_area'] == brain_area]
    return areas_clean
 

def orientation_extractor(unit_key, fpd):
    ''' This function allows to extract the mean activity corresponding to each repeat
    of the direction shown as part of the Monet2 stimuli for a specified neuron

    Parameters:
    unit_key: dictionary specifying the value for the session, scan_idx and unit_idx keys
    fpd: frames per direction of the movie shown (might change in different session, scan_idx pairs)
    
    Returns:
    df: DataFrame with a columnd showing the directions inn degrees, teh directionns in radians and the
    mean activity of that cell across all the Monet2 trial that had that direction
    '''
    
    #Select the movie trials in the appropiate session and scan
    trial_key = {'session': unit_key['session'], 'scan_idx': unit_key['scan_idx']}
    trial_info = nda.Trial & trial_key
    
    #Extract the direction variables for the Monet2 stimuli in these trials, matrix of shape [ntrials]
    #within it each item is a matrix of shape [1, ndirs]
    dirs = (trial_info* nda.Monet2).fetch('directions')
    spike_trace = (nda.Activity() & unit_key).fetch1('trace')

    #Extract start and end frames these have the same shape as the dirs matrix, ntrials
    #each item is the start or end id
    s = (trial_info * nda.Monet2).fetch('start_idx')
    e = (trial_info * nda.Monet2).fetch('end_idx')
    
    #Loop thorugh them and calculate average activity and corresponding direction
    directions=[]
    m_act = []
    trial_id = []
    d = 0

    #loop though each monet trial
    for seg in range(len(s)):
        st = s[seg]
        en = e[seg]
        c = 0

        #extract the relevant spike trace segment
        sp_red = spike_trace[st:en+1]
        #loop though each x (depends on fpd) frames with same orientation
        if fpd>6: #change number of frame that have the same direction since session 9 had a higher frame rate of 8
            for i in range(0, sp_red.shape[0], fpd):
                if c < 16:
                    directions.append(dirs[d][0][c]) #d: monet trial number, 0: selects array, c: orientation index
                    m_act.append(np.mean(sp_red[i:i+fpd])) #append mean activity to frames with same orientation
                else:
                    continue
                    #directions.append(dirs[d][0][c])
                    #m_act.append(np.mean(sp_red[i:]))
                c+=1
                trial_id.append(d)
        else:    
            for i in range(0, sp_red.shape[0], 6):
                if c < 16:
                    directions.append(dirs[d][0][c]) #d: monet trial number, 0: selects array, c: orientation index
                    m_act.append(np.mean(sp_red[i:i+6])) #append mean activity to frames with same orientation
                else:
                    directions.append(dirs[d][0][c])
                    m_act.append(np.mean(sp_red[i:]))
                c+=1
                trial_id.append(d)
        d+=1
    #Save them in a data frame  
    df = pd.DataFrame({'orientation':directions, 'mean_activity':m_act, 'trial_id':trial_id})
    
    #Turn orientation in to radians
    df['radians'] = df['orientation']*(np.pi/180)
    
    return df

def von_mises(theta, A, phi, k):
    '''Function describing the tuning curve of the neurone to the orientation of the stimulus if the neurone is ONLY
    orientation selective
    
    Parameters:
    theta: orientation of the stimulus
    A: amplitude of the cosine wave
    phi: offset of the wave
    rmax: maximum value the activation can take (max value of sigmoid function)
    L0: inflection point of the sigmoid
    
    Returns:
    activity of neuron at specified angle'''
    
    r = A*np.exp(k*(np.cos(2*(theta-phi))-1))
    
    return r

def von_mises_single(theta, A, phi, k):
    ''' Function describing the tuning curve of the neurone to the orientation of the stimulus if the neuron
    is both orientation and direction selective
    
    Parameters:
    theta: orientation of the stimulus
    A: amplitude of the cosine wave
    phi: offset of the wave
    rmax: maximum value the activation can take (max value of sigmoid function)
    L0: inflection point of the sigmoid
    
    Returns:
    activity of neuron at specified angle'''
    
    r = A*np.exp(k*(np.cos((theta-phi))-1))
    
    return r

def is_selective(df, max_rad, single = True):
    '''This function utilises a wilcoxon test to understand if there is a significant difference
    between the activity of a neuron at its estimated preferred orientation and its estimated least preferred
    orientation so as to understand if its oreintation and direction selective or just orientation
    
    Parameters:
    df: DataFrame with directions and mean activity at each direction
    max_rad: integer or float with estimated preferred orientation of the cell
    single: boolean idenfiying whether we are testing for oreintationn selectivity (single = True)
    or orientation and direction selectivity (single = False)
    
    Returns:
    statw: wilcoxon statistics value
    pw: p value of the wilcoxon test
    min_rad: estimated least preferred orientation
    '''
    #If there is a single peak, frequency of 2pi
    if single:
        if max_rad>np.pi:
            min_rad = max_rad-np.pi
        else:
            min_rad = max_rad+np.pi
    
    #If there are two peaks, frequency of pi
    else:
        if max_rad>(np.pi*1.5):
            min_rad = max_rad-(np.pi/2)
        else:
            min_rad = max_rad+(np.pi/2)
        
    
    closemax= df.iloc[(df['radians']-max_rad).abs().argsort()[:1]].iloc[0,3]
    closemin = df.iloc[(df['radians']-min_rad).abs().argsort()[:1]].iloc[0,3]
    max_act = df[df['radians'] == closemax]['mean_activity']
    min_act = df[df['radians'] == closemin]['mean_activity']
    statw, pw = wilcoxon(max_act, min_act)
    
    return statw, pw, min_rad