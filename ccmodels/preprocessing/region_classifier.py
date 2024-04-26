import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import sys

sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
#from ccmodels.preprocessing.utils import layer_extractor
import ccmodels.preprocessing.utils as utl
#from ccmodels.preprocessing.connectomics import load_table, client_version
import ccmodels.preprocessing.connectomics as conn
from standard_transform import minnie_transform_vx

def train_classifier(do_test=True, prepath="data/"):
    #Get the transform
    tform_vx = minnie_transform_vx()

    #Load the table of functionally matched neurons
    funcmatched = conn.read_table("functionally_matched", prepath=prepath) 

    #Drop unlabelled neurons
    funcmatched = funcmatched[funcmatched['pt_root_id']!=0]
    #Drop neurons recorded in more than one scan
    funcmatched = funcmatched.drop_duplicates(subset='pt_root_id', keep = 'first')

    #funcm = funcmatched[['id','pt_root_id',	'session','scan_idx','unit_id',	'pt_position_x', 'pt_position_y', 'pt_position_z']]
    funcm = funcmatched[['id','pt_root_id',	'session','scan_idx','unit_id',	'pt_position']]


    #Get the neurons for which we now the area
    brainarea = conn.read_table("area_membership", prepath=prepath)
    funcm_ba = brainarea.merge(funcm, on = ['session','scan_idx','unit_id'], how = 'inner')

    #Classes that we have to classify
    target_classes = funcm_ba["brain_area"].unique()

    #Encode the classes
    encoder = LabelEncoder()
    encoder.fit(target_classes)

    #Define features and outputs
    #Pass to list and then to array allow to unfold pt_position as a Nx3 numpy array, as we need
    x = np.array(list(funcm_ba["pt_position"]))
    y = encoder.transform(funcm_ba['brain_area'])


    #Potentially do a test with train-test split to convince ourselves this is actually working
    if do_test:
        #Split in train and test, and produce a fit
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        forest = RandomForestClassifier()
        forest.fit(x_train, y_train)

        forestpreds = forest.predict(x_test)
        print(classification_report(y_test, forestpreds, target_names=encoder.classes_))

    #Ise the full dataset for training to get better results in the real life
    forest = RandomForestClassifier()
    forest.fit(x, y)

    return forest, encoder



def predict(table, classifier, encoder):
    neuron_positions = np.array(list(table["pt_position"]))
    regions_idx = classifier.predict(neuron_positions)
    return encoder.inverse_transform(regions_idx)