# Introduction

This repository contains code to download and process connectivity data from the Microns project, which is used to implement connectome constrained models and analyse orientation selectivity in mouse V1. 

It's not just replicability: the `ccmodels` pakcage has several utilities that can help the user to deal with the complexities of the Microns dataset. We have designed these for our analysis, but we are sure they will be of help for other researchers. Here we explain how the inner parts of the package work and how the user can benefit from our previous effort. For this reason, the code (should be) documented enough for you to use it too!

If you use this code in your research, we just ask to cite the following publication:

> TODO write a great paper


For any problem regarding the code, please contact us.

## Dependencies 

We rely only on standard scientific Python packages, such as `numpy`, `pandas` or `scikit` (which can be obtained through the Conda distribution). For replicating our paper or using our preprocessed data there is no more dependencies. However, if you want to re-download or re-analyse new data you'll need 

- [CAVEclient](https://github.com/CAVEconnectome/CAVEclient) to query the connectomics data.
- [Standard transform](https://pypi.org/project/standard-transform/) to transform neurons position's coordinates. 
- Configure the [Microns NDA](https://github.com/cajal/microns-nda-access) to access functional data.

and their potential dependencies. We recommend installing packages in a virtual environment for safety. Note that we provide pre-processed data to replicate our analysis without having to query the database, which can be complicated. 

## Install

Just `git clone` this repository. We will refer to the `con-con-models/` folder as the _base_ directory. If you wish, you can install the package in editable mode by being in base directory and running:

`pip install -e` 

This is not required, but allows one to import the code from the `ccmodels` pakcage in a simpler way. 


# Replication Analysis 

## Repository structure

The repository has several folders with useful information. These are, 

- `ccmodels`: the main package, containing the base code you will new to query the APIs, clean the data, analyse it, and run our synthetic model. We'll go back to this later.

- `notebooks`: potentially useful Jupyter notebooks. 

- `scripts`: contains scripts that use the `ccmodels` package to perform several tasks. The scripts in the `data_download` subfolder do query the API and download all necessary tables. Scripts in `data_preprocessing` do process the downloaded data to generate clean tables, which are ready to use. Finally, `figures` contains scripts to do the analyses and generate figures. 

> **Important note:** the files `area_membership.py`, `funcdata_extraction.py`, and `sampler.py` have not yet been reviewed, since they deal with functional data. They mix download and preprocessing. This will be changed in the short term. 

- `data`: a folder containing all tables necessary for the analysis. `raw` contains tables (almost) as they come from queries to API, with a minimum preprocessing (deletion of duplicates, invalid values, etc.). The folder `in_processing` is used to store intermediate tables that are necessary for the package to work and should be of little interest to the user. Finally, the folder `preprocessed` contains the unit, connections and activity table for L2/3/4 of V1, ready to be read and used. We recommend start from thoseo

## Data structure

### Raw

- `aibsct.csv`: this table corresponds to `aibs_soma_nuc_metamodel_preds_v117`, keeping only key coluumns. All `pt_root_id = 0` have been filtered out. Duplicates in the `target_id` have been removed. 

- `coarsect.csv`: this table corresponds to `baylor_log_reg_cell_type_coarse_v1`. Same treatment as above. 

- `finect.csv`: this table corresponds to `baylor_gnn_cell_type_fine_model_v2`. Same treatment as above. 

- `neuronref.csv`: this table corresponds to `nucleus_ref_neuron_svm`, keeping only key coluumns. All `pt_root_id = 0` have been filtered out.

- `functionally_matched.csv`: this table corresponds to `coregistration_manual_v3`, keeping only key columns. All `pt_root_id = 0` have been filtered out. Duplicates in the `pt_root_id` have been removed.

- `proofreading.csv`: this table corresponds to `proofreading_status_public_release`, keeping only key columns. All `pt_root_id = 0` have been filtered out. All entries not fulfilling `pt_root_id == valid_id` have been filtered out. 

- `area_membership.csv`: contains to which area the functionally matched neurons do belong to.

> (Victor's note: I haven't yet manually replicated what generates the area memberships, so I don't know what potential limitations the table has)

### In processing

These are files which are in the middle of the processing and are useful, but not quite the final stage. 


- `orientation_fits.pkl`: contain the results of the orientation fits for each functionally matched neuron.

### Preprocessed

- `unit_table.csv`: contains a list of all L2/3/4 neurons in V1 that we were able to classify. It has neuron's `pt_root_id` and their respective properties, such as cell type, proofreading, or layer. When read using `load_tables` (see below), all `pt_root_id` are remapped from `0` to `n_neurons-1` and the column index is changed to just `id`. Note that in this tables functionally matched neurons (5067 of them) apear first.

- `connections_table.csv`: list of all connected neurons, giving weighted connections from `pre_id` to `post_id`.  When read using `load_tables` (see below) all the ids are remapped to match the unit table.

- `activity_table.csv`: this table contains all the activity for each functionally matched neuron and stimulus orientation. When it is read from `load_tables`, it is reshaped as `n_funcmatched_neuron X nangles` matrix, to facilitate algebra. Note that the i-th row of this table corresponds to neuron `i` in the unit table (and that's why all functionally matched neurons appear first there!).  

## Pipeline 

Here we explain how to replicate some of the common pipelines, assuming all dependencies are installed, and user has moved to the base directory. 

### Download functional data, fit tuning

> **TODO**, as this part of the pipeline needs to be double-checked. The information in this section comes from Jacopo's previous README and should still hold.

1. funcdata_extraction.py: to carry out the first data extraction and preprocessing step required to estimate the selectivity of each of the neurons under study and extract all the features required to carry out the thresholding criteria needed to determine more granularly the selectivity of neurons. The output of this file is a 'orientation_fits.pkl' file.  (NOTE: this is a time intensive step that requires both setting up access to the functional database and x hours to run).

2. selectivity_estimation.py: to assigne to each neuron whether it is elective or not, together with other useful characteristics utilised in our analises. The output of this script is a csv file called 'unit_table.csv' in a convenient format for analysis. 
NOTE: requires the output of funcdata_extraction.py to run.

3. responses_organizer.py: to organise the responses of the neurons under study and the corresponding angle shown in a format useful for subsequent analysis. The output is a csv file called 'activity_table.csv'.
NOTE: requires the output of funcdata_extraction.py to run.

### Generating connectomic tables 

1. `python scripts/data_download/connectome_tables.py` to download the tables related to neuron's information (v661).
2. `python scripts/data_preprocessing/generate_units_table.py` to generate the unit table. 
3. `python scripts/data_download/connectome_constructor.py` to download the synapses between the neurons in the newly generate unit table. Observe that the download happens in parts, so if it is interrupted, it can be restarted by modifying the `start_index` parameter in the downloader call.
4. `python scripts/data_preprocessing/merge_connc_parts.py` merges all the parts and writes the final connectivity table.

### Generating figures

The scripts expect to know the folder to save the figure after execution. Hence, 

`python scripts/figures/figX.py output_figure`

will produce `output_figure/figX.pdf`. 

> For the new data I have checked that this works for figs 2 and 3. 

> Figures might complain about not having Helvetica fonts installed, but the code will run the same. 


# Package overview

Now, we review the contents of the package and what are the modules of interest for the final user. The scripts have examples on what can be done, but we explain it here for better clarity.

In order to use the module


## Structure

The `ccmodels` is composed by several modules. 


> **Note:** for running code, you need to either install the package in editable mode (see above) or add `ccmodels` folder to your path. If you are not willing to do so, it's also possible to add the following **before** importing any of the `ccmodels` modules:
> ```python
> import sys
> sys.path.append("path/to/base/folder")
> ```
> This line adds the package to the path temporarily. 

### Preprocessing

The first one, `preprocessing`, has utilities dedicated to download the data and generate the tables for final use. Most users would not want to mess with this and would do better directly jumping to the analysis given the preprocessed data that we provide. The most relevant thing of this module might be the `rawloader` file, which allows one to quickly load all the raw/semiprocessed data just by knowing the name of the desired table. 

```python

import ccmodels.preprocessing.rawloader as rawl 

fm = rawl.read_table("aibs", prepath="path/to/data/") 
```

The format is not needed. The `prepath` variable is used to indicate where the `data` folder is. It is not needed if the Python files are executed from the base folder, but if you are working in a notebook `basefolder/notebooks/mynotebook.ipynb` then you'll need `prepath=../data/`. 
Other interesting files inside the module are the `region_classifier.py`, which allow you to train a classifier to predict the brain region of the selected neurons, or the `connectomics.py` file that merges all the tables.

### Data Analysis

**The `dataanalysis` module contains some of the most useful files for the end user.**  First of all, there is the `processedloader`file. Analogously to the `rawloader`, takes care of reading the data. However, `processedloader` loads the three main elements of the processed data.


```python

import ccmodels.dataanalysis.processedloader as loader 

units, connections, activity = loader.read_tables(orientation_only=True, prepath="path/to/data")
```

**The function `read_tables `is fundalmental**, because it does not only read the data, but also remaps all the `pt_root_id` indices to go from `0` to `n_neurons-1` instead. Additionally, formats `activity` as a `n_funcmatch_neurons X nangles` matrix, and computes the angle difference in the connectivity table. All angles and angle differences are constrained accordingly with the `orientation_only` parameter, that allow us to use all 16 angles (if it's `False`) or detect only orientation (if it's `True`, as default). 

Once that we have loaded our data, a very common task is to **filter** it correctly. For this, we have the `filters` file, which contain several functions which are very handy to filter complex cases in one line. For neurons, for example, 

```python
import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl

units, connections, activity = loader.read_tables(orientation_only=True, prepath="path/to/data")

#Units in L23
units23 = fl.filter_neurons(units, layer="L23")

#Inhibitory neurons
units_inh = fl.filter_neurons(units, cell_type="inh")

#Both L23 and inhibitory neurons
units_inh = fl.filter_neurons(units, layer="L23", cell_type="inh")

#Tuned neurons with quality proofread
units_t = fl.filter_neurons(units, tuning="tuned", proofread="decent")
```

The most relevant filter is the `tuning` one. It can take several values, `'tuned'` (functionally matched neurons that are either orientation or direction selective), `'untuned'` (not selective neurons), `'matched'` (neurons that have been functionally matched, including non selective ones) and `'unmatched'` (non-functionally matched). Additionally, the `'proofread'` parameter uses [indications from Allen's institute on proofreading](https://allenswdb.github.io/microns-em/em-background.html) to indicate the level of proofread of the neurons, having `'minimum'`, `'decent'`, `'good'` and `'perfect'`. The `'minimum'` category stands for clean axons, and `'decent'` for extended axons. The `'decent'` level should give close to complete connectivity already. The next two levels include dendrite proofreading. 


Selecting connections is also easy with `filters`. One way to do it is from the function `synapses_by_id`, which allows us to get the connections whose presynaptic or postsynaptic neuron is in a list of given ids. The shortcut function `filter_connections` helps for the most common tasks.

```python
#Get ids of neurons in layer L23, and then check presynaptic neurons in this list...
units23 = fl.filter_neurons(units, layer="L23")
ids23 = units23["id"]
conn23 = fl.synapses_by_id(ids23, connections, who="pre") 

#A simpler way to do the above is
conn23 = fl.filter_connections(units, connections, layer="L23", who="pre")

#Get all connections where neuron has ids 7, 77 or 777, no matter if its presynaptic or postsynaptic
conn7 = fl.synapses_by_id([7,77,777], connections, who="any") 

#Check connections between neurons which are both tuned
conn_tuned = fl.filter_connections(units, connections, tuning="tuned", who="both")
```

The `who` parameter indicates if either `'pre'` or `'post'` synaptic neurons are affected by the filtering. It is possible to demand that `'both'` neurons fulfill the filter, or that `'any'` of them does.

> Notice that there is no way to apply different conditions on presynaptic and postsynaptic neurons, for now. So if one would like to have inhibitory presynaptic neurons and excitatory postsynaptic ones, it goes as 
>```python
>conn_preinh = fl.filter_connections(units, connections, cell_type="inh", who="pre")
>conn_postexc= fl.filter_connections(units, conn_preinh, cell_type="exc", who="post")
>```

These are probably some of the most useful functions on the codebase. Then, `statistics_extraction` has functions devoted to obtain the connection probabilities between different kinds of neurons, while `currents` computes the input synaptic current in different scenarios. The functions in those files are called by Figures 2 and 3, respectively.


### Model analysis

This folder contains three files, `matrixsampler`, `model` and `utils`. The first one is used to sample matrices with a connectivity pattern similar to the one extracted from the data. The `model` file contains several functions that define the model and integrate the differential equation, while `utils` just holds some of the mathematical definitions. 

The most important function is undoubtely `make_simulation(k_ee, p_ee, J, g)` that given an average connectivity `k_ee`, a connection probability `p_ee`, a coupling stregnth `J` and inhibitory scaling `g` makes a single simulation of the model, returning the timeseries of all neurons, as well as the rates per each layer. 

> Note that these functions have not yet been tested with the new version of the dataset, but they should work anyway (maybe with some tuning of the paths to the files, which are now in `data/data_old/sample_data.pkl`)


### Utils

General utils for the model. The `angleutils` are found here. These contain all the necessary code to deal with the angles through the entire project, doing tasks as constraining the angles in the corresponding range, moving from the interval `[0, 2π[` to `[-π, π[`, and so on. 


### Plotting

Finally, the plotting folder contains files dedicated to keep a consistent styling and good practices for figure-quality papers. The `color_reference` keeps the color code (in order to be able to change in one go, e.g., the color that represents L23 and L4 in all figures) for all the figures, while the `styles` sets the figure size to fit one- or two-columns sizes in mm, sets Helvetica and font sizes, changes legends style, etc. The styling function is just from [here](https://allenswdb.github.io/microns-em/em-background.html).

> All paper figures should start by sty.master_format() to apply the correct formating and have size sty.one_col_size or sty.two_col_size. The `styles` file also has convenient functions, such as `despine`, or `label_axes`, which automatically applies (a), (b), (c)... labels to all panels. However, the most important thing is to ensure style coherence and correct figure size.

