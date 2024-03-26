''' This script fetches the area membership table and saves it as is in a csv file. 
    Note: requires access to the functional database.'''

import pandas as pd
from microns_phase3 import nda, utils

areas = nda.AreaMembership.fetch(format='frame').reset_index()
areas.to_csv('../../data/raw/area_membership.csv')