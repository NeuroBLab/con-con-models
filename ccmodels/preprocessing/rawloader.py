import pandas as pd

def read_table(name, prepath="data/", splitpos=False):
    """
    Read an already stored raw/in-processing data table. 
    For loading the preprocessed data, please refer to the datanalysis module.
    """

    #Get the path to the table and read it
    path = prepath + _get_table_path(name)

    if "csv" in path:
        table = pd.read_csv(path)
    elif "pkl" in path:
        table = pd.read_pickle(path) 

    #Merge together the positions, so the column contains a list with 3 values
    if "pt_position_x" in table.columns and not splitpos:
        splitpos = ["pt_position_x", "pt_position_y", "pt_position_z"]
        table["pt_position"] = list(table[splitpos].values)  
        table.drop(columns=splitpos, inplace=True)

    #Return the thing
    return table

def _get_table_path(name):
    """
    Not intended for use outside of the module. 
    Auxiliary function that indexes in which folder each file is, and return its path. 
    """
    folder_content = {"in_processing" : ["ei_table.csv", "orientation_fits.pkl"], 
                      "raw" : ["aibsct.csv", "area_membership.csv", "coarsect.csv", "finect.csv", 
                               "functionally_matched.csv", "neuronref.csv", "proofreading.csv"]}

    #Check if the name is in one of the filenames...
    for key in folder_content:
        files_in_folder = folder_content[key]
        for filename in files_in_folder:
            #If it is, just return the path
            if name in filename: 
                return f"{key}/{filename}"