#This simple script downloads all the necessary tables using CAVEclient. 
#User needs to have a working CAVEClient configuration.
import sys
import os 
sys.path.append(os.getcwd())

import ccmodels.preprocessing.downloader as down 

client = down.get_client(661)
down.download_tables(client)