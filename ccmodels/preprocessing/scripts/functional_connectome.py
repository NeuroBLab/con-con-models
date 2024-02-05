'''
By running this script you can extract a pickle file containing a subset of teh connectome
with the connnectivity amongst all functionally matched neurons from L2/3/4 of V1.
In addition pre and post synpatic neurons also contain information on...

Estimated runtime: 1h 45 minutes
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from caveclient import CAVEclient
from standard_transform import minnie_transform_vx
from ccmodels.preprocessing.extractors.utils import connectome_constructor, subset_v1l234


def main():
    #define Caveclient and database version
    client = CAVEclient('minnie65_public')
    client.materialize.version = 661

    #Extracting all the root id of the functionally matched cells
    funct_match = client.materialize.query_table('coregistration_manual_v3')

    fun = np.array(list(set(funct_match[funct_match['pt_root_id'] != 0]['pt_root_id'])))

    #Extract connectome
    connect_functional = connectome_constructor(client, fun, fun, 500)

    #Save it
    connect_functional.to_csv('functional_connectomev661.csv', index = False)


if __name__ == '__main__':
    main()

