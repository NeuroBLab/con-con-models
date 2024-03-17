# Introduction
Welcome this repository contains the code to implement connectome constrained models and analyse orientation selectivity in mouse V1 as detailed in the paper:
[paper title and link]

Below you will find a breakdown of how the code is structured, tutorials on how to run the scripts to replicate the analsysis done in the paper and how to install our package so that you can use it on your machine locally. We hope everything is clear and we are open to comments and suggestions if you have any!

# Install package in editable mode
Once you have cloned the repository we advise starting a new virtual environment in the root directory (con-con-model) and installing all the required packages from the requirements.txt file.
To be able to utilise the package you then need to install it in editable mode in your virtual environment by being in the root directory of the project and running:

    pip install -e .

# Reproducing our analyses

## Preprocessing the data
To follow our steps you can run:
    1. funcdata_extraction.py: to carry out the first data extraction and preprocessing step required to estimate the selectivity of each of the neurons under study and extract all the features required to carry out the thresholding criteria needed to determine more granularly the selectivity of neurons. The output of this file is a 'orientation_fits.pkl' file.  (NOTE: this is a time intensive step that requires both setting up access to the functional database and x hours to run).

    2. selectivity_estimation.py: to assigne to each neuron whether it is elective or not, together with other useful characteristics utilised in our analises. The output of this script is a csv file called 'unit_test.csv' in a convenient format for analysis. 
    NOTE: requires the output of funcdata_extraction.py to run.

    3. responses_organizer.py: to organise the responses of the neurons under study and the corresponding angle shown in a format useful for subsequent analysis. The output is a csv file called 'activity.csv'.
    NOTE: requires the output of funcdata_extraction.py to run.

    4. connectome_constructor.py: to extract the subest of the connectome containing the neurons under study. The ouptut is a csv file called 'connections.csv'. NOTHE: requires the output of selectivity_estimation.py to run.

## Generate the figures
### Generate one of the figures
Once the package has been installed simply run the script for the figure you wish to generate:
    
    python3 figure_scripts/fig1 w h save_path

    where:  'w' is the desired figure's width (we used x),
            'h' is the desired figure's height (we used y)
            'save_path' is the path of the location you desire to save the figure in, ending with a '/'.

### Generate all of the figures

# Run the con-con-model