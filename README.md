# Introduction
Welcome this repository contains the code to implement connectome constrained models and analyse orientation selectivity in mouse V1 as detailed in the paper:
[paper title and link]

Below you will find a breakdown of how the code is structured, tutorials on how to run the scripts to replicate the analsysis done in the paper and how to install our package so that you can use it on your machine locally. We hope everything is clear and we are open to comments and suggestions if you have any!

# Install package in editable mode
Once you have cloned the repository we advise starting a new virtual environment in the root directory (con-con-model) and installing all the required packages from the requirements.txt file.
To be able to utilise the package you then need to install it in editable mode in your virtual environment by being in the root directory of the project and running:

    pip install -e .

# Generate one of the plots
Once the package has been installed simply run the script for the figure you wish to generate:
    
    python3 figure_scripts/fig1 w h save_path

    where:  'w' is the desired figure's width (we used x),
            'h' is the desired figure's height (we used y)
            'save_path' is the path of the location you desire to save the figure in, ending with a '/'.

