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

def min_act(max_rad, model_type, dirs):
    '''This function returns the oreintation for where the minimum of the selective activity should be
    
    Parameters:
    max_rad: integer or float with estimated preferred orientation of the cell
    model_type: string idenfiying whether the modelled cell is  oreintationn selectivity (model_type = 'single')
    or orientation and direction selectivity (model_type = 'double')
    
    Returns:
    min_rad: estimated least preferred orientation
    '''
    #If there is a single peak, frequency of 2pi -> neuron is direction selective
    if model_type == 'direction':
        if max_rad>np.pi:
            min_rad = max_rad-np.pi
        else:
            min_rad = max_rad+np.pi
    
    #If there are two peak, frequency of pi -> neuron is orientation selective
    # NOTE: here we treat those neurons that are not selective as orientation selective 
    # for the purpose of calculating an osi value also for them

    else:
        if max_rad>(np.pi*1.5):
            min_rad = max_rad-(np.pi/2)
        else:
            min_rad = max_rad+(np.pi/2)

    
    ind_min = np.argmin(np.abs(dirs- min_rad))

    closemin = dirs[ind_min]
    return closemin

def constrainer(dirs, reversed = False):
    '''Function that constrains given matrix of directions between [-2pi, 2pi] in to (-pi, pi]
    
    Parameters:
    dirs: numpy array of directions
    
    Returns:
    all_truncated: numpy array of constrained directions
    '''
    

    #remap between [-np.pi, np.pi]
    #find cells below -np.pi
    smaller = (dirs<=-np.pi).astype(int)*(2*np.pi)
    
    #find cells above np.pi
    larger = (dirs>np.pi).astype(int)*(2*np.pi)

    #add 2pi to dirs below -np.pi
    small_truncated = dirs+smaller

    #subtract 2pi to cells above np.pi
    all_truncated = small_truncated-larger

    if reversed:
        smaller = (dirs<0).astype(int)*(2*np.pi)
        detruncated = dirs+smaller
        return detruncated

    return all_truncated

def constrain_act_range(post_root_col, post_root_id, directions, pre_df, currents = True):
    '''This function maps the discretized directions shown in the stimulus from the [-2pi, 2pi]
    range to the [-pi, pi] range and re-orders the activities of each pre-synaptic
    connections of a specified post-synaptic cell according to the new direction mapping
    
    Parameters:
    post_root_col: str, column containing postsynaptic ids of neurons
    post_root_id: id of the post_synaptic cell
    directions: array of discretized directions in [-2pi, 2pi] range
    pre_df: data frame containing activities of pre-synaptic cell and key (post_root_id) specifiying which post_synaptic cell they connect to 
    
    Returns:
    reordered_act: list where each item is an array of the activity for a pre_synaptic cell
    with values reordered according to their new [-pi, pi] range

    constrained_dirs: list of directions remapped in range (-pi, pi]
    '''

    #select all pre synaptic cells
    cell = pre_df[pre_df[post_root_col] == post_root_id]
    
    #differences with post max
    arr_diffs = directions-cell['post_po'].values[0]

    #constrainn directions between (-pi, pi]
    all_truncated = constrainer(arr_diffs)
    all_truncated = np.around(all_truncated, 6)
    all_truncated[all_truncated ==-3.141593] = 3.141593

    #extract index sorted from smallest direction to largest
    idx= np.argsort(all_truncated)
    
    #generate array with activities of pre_synaptic cells
    if currents:
        activities = np.array(cell['current'].tolist())
    else:
        activities = np.array(cell['pre_activity'].tolist())


    #order these activities accoridng to their sorted value in the new
    #[-np.pi, np.pi] range
    reordered_act = list(activities[:,idx])

    constrained_dirs = list(all_truncated[idx])


    return reordered_act, constrained_dirs



def layer_extractor(input_df, transform, column = 'pre_pt_position'):
    input_df['pial_distances'] = transform.apply(input_df[column])

    #Use the y axis value to assign the corresponding layer as per Ding et al. 2023
    layers = []
    for i in input_df['pial_distances'].iloc[:]:
        if 0<i[1]<=98:
            layers.append('L1')
        elif 98<i[1]<=283:
            layers.append('L2/3')
        elif 283<i[1]<=371:
            layers.append('L4')
        elif 371<i[1]<=574:
            layers.append('L5')
        elif 574<i[1]<=713:
            layers.append('L6')
        else:
            layers.append('unidentified')

    input_df['cortex_layer'] = layers   
    return input_df