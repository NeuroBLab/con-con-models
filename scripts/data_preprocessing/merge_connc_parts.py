import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.preprocessing.connectomics as conn

#Just call the merger.
synapses = conn.merge_connection_tables()
synapses.to_csv("data/preprocessed/connections_table.csv")
