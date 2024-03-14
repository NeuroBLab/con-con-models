#Imports
import numpy as np

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

if __name__ == '__main__':
    import os
    print(os.getcwd())