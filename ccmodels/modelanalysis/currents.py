import sys
sys.path.append(".")

import numpy as np
import pandas as pd

import ccmodels.dataanalysis.currents as cur
import ccmodels.dataanalysis.utils as utl
import ccmodels.dataanalysis.filters as fl
import ccmodels.utils.angleutils as au

def get_model_prefori(units_sampled, rates_sampled, vij):

    #Get tuned postsynaptic neurons
    sampled23 = fl.filter_neurons(units_sampled, layer='L23', tuning='tuned')

    #Make the untuned rates go to the average
    rates_sampled = utl.get_untuned_rate(units_sampled, rates_sampled) 

    #Get the currents from presynaptic to postsynaptic
    currents = cur.get_currents_subset(units_sampled, vij, rates_sampled, pre_ids=units_sampled['id'], post_ids=sampled23['id'])

    #Estimated preferrred orientation
    return np.argmax(currents, axis=1) 


def bootstrap_mean_current(units, vij, rates, tuning=['matched', 'matched'], cell_type=['exc', 'exc']):
    neurons_L4  = fl.filter_neurons(units, layer='L4', tuning=tuning[0], cell_type='exc')
    neurons_L23 = fl.filter_neurons(units, layer='L23', tuning=tuning[0], cell_type=cell_type[0])

    neurons_L23_post = fl.filter_neurons(units, layer='L23', tuning=tuning[1], cell_type=cell_type[1])

    currents = {}

    currents['L4']    = cur.get_currents_subset(units, vij, rates, post_ids=neurons_L23_post['id'], pre_ids=neurons_L4['id'], shift=True)
    currents['L23']   = cur.get_currents_subset(units, vij, rates, post_ids=neurons_L23_post['id'], pre_ids=neurons_L23['id'], shift=True)

    currents['Total'] = currents['L23'] + currents['L4']

    return currents