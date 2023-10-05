import os
import statistics as s
from itertools import product
import json
import numpy as np

data = json.load(open('../../src/data/val_set.json'))

ratios = ['WH', 'WD', 'HD']

def calc_ratio(dims, reference=0):
    ''' Calculate the ratio between the three dimensions. Setting the first 
    dimension as reference value. Returns tuple of size 3 '''
    ratio = []
    for dim in dims:
        ratio.append(dim / dims[reference])
    
    return ratio


def calculate_ratios(data):
    ''' For all the boxes in the data, 
    calculate the ratio of each dimension and find the mean and std of the ratios. '''
    ratios = []
    for img in data:
        for box in data[img]:
            xyz = box[3:]
            ratio = calc_ratio(xyz)
            ratios.append(ratio)

    return np.array(ratios)

def calc_avgs(data):
    ''' Calculate the mean and std of the ratios of each dimension. '''
    ratios = calculate_ratios(data)
    means = []
    stds = []
    for i in range(3):
        means.append(s.mean(ratios[:, i]))
        stds.append(s.stdev(ratios[:, i]))
   
    return means, stds


def summarize(data):
    ''' Calculate combinations using means and stds. '''
    means, stds = calc_avgs(data)

    x_vals = [means[0] - stds[0], means[0], means[0] + stds[0]]
    y_vals = [means[1] - stds[1], means[1], means[1] + stds[1]]
    z_vals = [means[2] - stds[2], means[2], means[2] + stds[2]]

    # Create xyz combinations of the values
    combinations = list(product(x_vals, y_vals, z_vals))
    combinations.append((1.0, 1.0, 1.0))

    return list(set(combinations))

xyz = summarize(data)
print(xyz)