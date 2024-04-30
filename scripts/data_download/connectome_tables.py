#This simple script downloads all the necessary tables using CAVEclient. 
#User needs to have a working CAVEClient configuration.
import sys

sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
import ccmodels.preprocessing.connectomics as conn
client = conn.client_version(661)
conn.download_tables(client)