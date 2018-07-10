#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 19:27:13 2018

@author: chuxuan
"""

import tensorflow as tf 
import model 
from dataset import DataSet,get_test_data
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt

coarse_params = {} # 定义一个新的dict
refine_params = {}
REFINE_TRAIN = True
FINE_TUNE = True
COARSE_DIR = "coarse"
REFINE_DIR = "refine"

if REFINE_TRAIN:
    for variable in tf.all_variables():
        variable_name = variable.name
        #print("parameter: %s" % (variable_name))
        if variable_name.find("/") < 0 or variable_name.count("/") != 1:
            continue
        if variable_name.find('coarse') >= 0:
            coarse_params[variable_name] = variable
        #print("parameter: %s" %(variable_name))
        if variable_name.find('fine') >= 0:
            refine_params[variable_name] = variable
else:
    for variable in tf.trainable_variables():
        variable_name = variable.name
        #print("parameter: %s" %(variable_name))
        if variable_name.find("/") < 0 or variable_name.count("/") != 1:
            continue
        if variable_name.find('coarse') >= 0:
            coarse_params[variable_name] = variable
        if variable_name.find('fine') >= 0:
            refine_params[variable_name] = variable


saver_coarse = tf.train.Saver(coarse_params)

if REFINE_TRAIN:
    saver_refine = tf.train.Saver(refine_params)
    
    
if FINE_TUNE:
    coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
    
    if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
        print(coarse_ckpt.model_checkpoint_path)
        print("Pretrained coarse Model Loading.")
        #saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
        #print("Pretrained coarse Model Restored.")
    else:
        print("No Pretrained coarse Model.")
        
    if REFINE_TRAIN:
        refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
        if refine_ckpt and refine_ckpt.model_checkpoint_path:
            print("Pretrained refine Model Loading.")
            #saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
            #print("Pretrained refine Model Restored.")
        else:
            print("No Pretrained refine Model.")

tf.reset_default_graph()
#with tf.Graph().as_default():
#TEST_FILE = "test.csv"
#img=mpimg.imread('stinkbug.png')
keep_conv = tf.placeholder(tf.float32)

images, depths, invalid_depths = get_test_data('test.csv')
coarse = model.inference(images,trainable=False)
logits = model.inference_refine(images,coarse,keep_conv,trainable=False)
loss = model.loss(logits, depths, invalid_depths)
init_op = tf.global_variables_initializer() #改了
sess = tf.Session()
sess.run(init_op)
loss_value,depths_estimation =sess.run([loss, logits],feed_dict={keep_conv: 0.8})
print(loss_value)
#img = mpimg.imread()
plt.imshow(depths_estimation[0,:,:,0],cmap='gray')
sess.close()
# 问题1：现在数据预测的结构总是在变。。。。。。。。。why？理解的不对么
# 问题2：
