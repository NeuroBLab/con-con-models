import numpy as np
from scipy.stats import sem
from fnn import microns

#Get an image of gratings and spatial frequency k=2πf. 
def gratings(x, k, theta, phi):
    u = np.array([np.cos(theta), np.sin(theta)])
    return np.cos(k*np.dot(x,u) + phi) 

#Convert the cosine stimuli to a 0-255 range
def to_uint8(x):
    return ((x+1) * 0.5 * 255).astype(np.uint8)

#Get the average response of the digital twin to moving gratings of different orientation and spatial frequency
def compute_DT_response(session, scan_idx, DT_directory, datapath = 'data'):
    #Load the Digital Twin
    model, ids = microns.scan(directory=DT_directory, session=session, scan_idx=scan_idx)

    #Dimensions of the (virtual) screen
    width, height = 256, 144

    #Spatial frequencies and orientations
    frequencies = np.arange(0.01, 0.17, 0.02)
    orientations = [j * np.pi / 8 for j in range(8)]

    #All pixels is a list of (x,y) with all pairs of pixels in the screen
    xp, yp = np.meshgrid(np.arange(0,width), np.arange(0,height), indexing='ij')
    allpixels = np.column_stack((xp.ravel(), yp.ravel()))
    xp, yp = None, None

    #Declare the arrays to store results
    average_responses = np.empty((len(frequencies), len(orientations), model.units))

    #Number of frames per second and duration of each stimuli
    fps = 30
    duration = 2 
    nframes = int(fps * duration)
    frames_gray = 15

    #Do the same for each spatial frequency
    for i,f in enumerate(frequencies):
        k = 2*np.pi*f

        #Do each orientation...
        for j,theta in enumerate(orientations):
            #Generate the gratings image in uint8

            stimulus = np.full([frames_gray, width, height], 127, dtype=np.uint8)
            for frame in range(nframes):
                phi =  (5 * 2*np.pi / nframes) * frame 
                g = gratings(allpixels, k, theta, phi).reshape(width,height)
                g = to_uint8(g)
                stimulus = np.concatenate((stimulus, g[np.newaxis, :, :]), axis=0)

            #Predict the responses
            responses = model.predict(stimuli=stimulus)

            #Get the average firing rate for this case. 
            average_responses[i,j,:] = responses[frames_gray:, :].mean(axis=0)

    np.save(f"{datapath}/in_processing/responses_gratings_dt/{session}_{scan_idx}.npy", average_responses)

#Read the results from the above function to get tuning curves
def get_tuning_curve(session, scan_idx, spfreqonly = True, datapath = 'data'):
    average_responses = np.load(f"{datapath}/in_processing/responses_gratings_dt/{session}_{scan_idx}.npy")

    #Average over frequencies -> orientation tuning curves, and viceversa
    #Return only what is requested, in many cases we will not need the oris
    if spfreqonly:
        #Return only aver_spa; shape = n_neurons x 8
        return np.transpose(average_responses.mean(axis=1))  
    else:
        aver_ori = np.transpose(average_responses.mean(axis=0)) 
        aver_spa = np.transpose(average_responses.mean(axis=1))  
        return aver_ori, aver_spa 
