#This simple script downloads all the necessary tables using CAVEclient. 
#User needs to have a working CAVEClient configuration.
import sys

sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
import ccmodels.preprocessing.downloader as down 

client = down.get_client(661)
down.download_tables(client)