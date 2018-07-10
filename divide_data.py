#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 19:34:32 2018

@author: chuxuan
"""
import random 
import os

def train_test_data(NumOfImage,NumOfTrainImge):
    all_=random.sample(range(NumOfImage),NumOfImage)
    a = all_[0:NumOfTrainImge]
    b = all_[NumOfTrainImge:len(all_)]
    
    trains = []
    tests = []
    
    for i in range(len(a)):
        train_image_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (a[i]))
        train_depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (a[i]))
        trains.append((train_image_name, train_depth_name))
    
    for i in range(len(b)):
        test_image_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (b[i]))
        test_depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (b[i]))
        tests.append((test_image_name, test_depth_name))
        
    with open('train.csv', 'w') as output:
            for (image_name, depth_name) in trains:
                output.write("%s,%s" % (image_name, depth_name))
                output.write("\n")
    
    with open('test.csv', 'w') as output:
            for (image_name, depth_name) in tests:
                output.write("%s,%s" % (image_name, depth_name))
                output.write("\n")
