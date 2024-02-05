'''
By running this script you can extract a pickle file containing a subset of teh connectome
with the connnectivity amongst all functionally matched neurons from L2/3/4 of V1.
In addition pre and post synpatic neurons also contain information on...

Estimated runtime: 44 minutes
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


    v1l234_neur = subset_v1l234(client, table_name = 'coregistration_manual_v3', area_df = 'con-con-models/data_full/v1_n.csv')

    #Extracting all the root id of the functionally matched cells
    nv1l234 = np.array(list(set(v1l234_neur[v1l234_neur['pt_root_id'] != 0]['pt_root_id'])))

    #Extract connectome
    connect_v1l234 = connectome_constructor(client, nv1l234, nv1l234, 500)

    #Save it
    connect_v1l234.to_csv('connectomev661_v1l234.csv', index = False)

if __name__ == '__main__':
    main()

