'''
By running this script you can extract a csv file containing a subset of the connectome
with the connnectivity amongst all functionally matched neurons from the specified connectomics
database version.

Estimated runtime: 1h 45 minutes
'''

#TODO
#add argparser option to specify where to read and write files to

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from caveclient import CAVEclient
from ccmodels.preprocessing.extractors.utils import connectome_constructor, subset_v1l234


def main():
    #define Caveclient and database version
    version = 661
    client = CAVEclient('minnie65_public')
    client.materialize.version = version

    #Extracting all the root id of the functionally matched cells
    funct_match = client.materialize.query_table('coregistration_manual_v3')

    fun = np.array(list(set(funct_match[funct_match['pt_root_id'] != 0]['pt_root_id'])))

    #Extract connectome
    connect_functional = connectome_constructor(client, fun, fun, 500)

    #Save it
    connect_functional.to_csv(f'functional_connectomev{version}.csv', index = False)


if __name__ == '__main__':
    main()

