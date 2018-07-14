#encoding: utf-8
#%%
#This is newly added comment
from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict,output_save
import model
import train_operation as op
import random
import os
import matplotlib.pyplot as plt

MAX_STEPS = 10000#10000000
LOG_DEVICE_PLACEMENT = False #??????


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

COARSE_DIR = "coarse"
REFINE_DIR = "refine"

NumOfImage = 18000
NumOfTrainImge = 14400
NumOfTest = NumOfImage - NumOfTrainImge
FINE_TUNE = True

def train(REFINE_TRAIN):
    BATCH_SIZE = 8
    
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        keep_conv = tf.placeholder(tf.float32)
        images, depths, invalid_depths,features = dataset.csv_inputs(TRAIN_FILE)
        
        if REFINE_TRAIN:
            print("refine train.")
            coarse = model.inference(images,trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv) #？？？这个 参数是什么
        else:
            print("coarse train.")
            logits = model.inference(images)
            
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.global_variables_initializer() #改了

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))# 不打印设备分配日志
        sess.run(init_op)    

        # parametersi
        coarse_params = {} # 定义一个新的dict
        refine_params = {}
        
        if REFINE_TRAIN:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" %(variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
                  
        # define saver
        print (coarse_params)
        saver_coarse = tf.train.Saver(coarse_params)
        
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
            
        # fine tune 微调。。。
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print(coarse_ckpt.model_checkpoint_path)
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            else:
                print("No Pretrained coarse Model.")
                
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                else:
                    print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        lossli=[]
        lossli1=[]
        for step in range(MAX_STEPS):
            index = 0
            lossli=[]
            print('-------------------------------')
            for i in range(3000): 
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8})  
                if i % 100 == 0:
                    print('[Epoch]:',step,'[iteration]:',i,'[Train losses]:',loss_value)
                lossli.append(loss_value)
                index += 1
            lossli1.append(np.mean(lossli))
            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        plt.figure()
        plt.plot(lossli1)
        plt.savefig("train_loss.jpg")    
        plt.xlabel("Epoch")
        plt.ylabel("Train_loss")
        plt.title("Train_Loss for Each Epoch")
        coord.request_stop() #请求所有线程停止
        coord.join(threads) #等待所有的线程完成
        sess.close()

def test():
    
    BATCH_SIZE = 1
    with tf.Graph().as_default():
        dataset = DataSet(BATCH_SIZE)
        keep_conv = tf.placeholder(tf.float32)
        images, depths, invalid_depths,features = dataset.csv_inputs(TEST_FILE)
        coarse = model.inference(images,trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv,trainable=False)
            
        loss1 = model.loss(coarse, depths, invalid_depths)
        loss2 = model.loss(logits, depths, invalid_depths)
        init_op = tf.global_variables_initializer() #改了
        

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))# 不打印设备分配日志
        sess.run(init_op)    
        
        coarse_params = {} # 定义一个新的dict
        refine_params = {}
        
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
    
        saver_coarse = tf.train.Saver(coarse_params)
        saver_refine = tf.train.Saver(refine_params)
            
        # fine tune 微调。。。
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)   
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                #print(coarse_ckpt.model_checkpoint_path)
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
            if refine_ckpt and refine_ckpt.model_checkpoint_path:
                #print(refine_ckpt.model_checkpoint_path)
                saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
        
        # test
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        index = 0
        ls1=[]
        ls2=[]
        print('\n','---------Examples---------:')
        for step in range(NumOfTest):
            #print('-----------------------------------------')
            loss_value1, loss_value2,logits_val,coarse_val ,images_val,features_ = sess.run([loss1,loss2, logits,coarse,images,features], feed_dict={keep_conv: 1})
            ls1.append(loss_value1)
            ls2.append(loss_value2)
            if step % 1 == 0:
                index = index + 1 
                print(features_,'Coarse losses:',loss_value1,'Refine losses:',loss_value2,'\n')
                output_save(coarse_val,logits_val,images_val,index,"data/test")
        ls1m=np.mean(ls1)
        ls2m=np.mean(ls2)
        print('---------Testing Results--------:')
        print('Coasre image mean losses:',ls1m)
        print('Refine image mean losses:',ls2m)
        coord.request_stop() #请求所有线程停止
        coord.join(threads) #等待所有的线程完成
        sess.close()

def train_test_data(NumOfImage,NumOfTrainImge):
    all_=random.sample(range(NumOfImage),NumOfImage)
    a = all_[0:NumOfTrainImge]
    b = all_[NumOfTrainImge:len(all_)]
    
    trains = []
    tests = []
    
    for i in range(len(a)):
        train_image_name = os.path.join("data", "hand_data", "%05d.jpg" % (a[i]))
        train_depth_name = os.path.join("data", "hand_data", "%05d.png" % (a[i]))
        trains.append((train_image_name, train_depth_name))
    
    for i in range(len(b)):
        test_image_name = os.path.join("data", "hand_data", "%05d.jpg" % (b[i]))
        test_depth_name = os.path.join("data", "hand_data", "%05d.png" % (b[i]))
        tests.append((test_image_name, test_depth_name))
        
    with open('train.csv', 'w') as output:
            for (image_name, depth_name) in trains:
                output.write("%s,%s" % (image_name, depth_name))
                output.write("\n")
    
    with open('test.csv', 'w') as output:
            for (image_name, depth_name) in tests:
                output.write("%s,%s" % (image_name, depth_name))
                output.write("\n")

def main(argv=None):
    
    if not gfile.Exists(COARSE_DIR): #如果不存在 新建一个文件夹叫（COARSE_DIR）
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    
    train_test_data(NumOfImage,NumOfTrainImge)
    print('--------Start Training-----------:')
    train(False)
    train(True)
    print('--------Start Testing-----------:')
    test()
    print('---------Finish---------')

if __name__ == '__main__':
    tf.app.run()
