#encoding: utf-8
#%%
from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op
import random
import os

MAX_STEPS = 100#10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"
COARSE_DIR = "coarse"
REFINE_DIR = "refine"

NumOfImage = 1000
NumOfTrainImge = 800

FINE_TUNE = True

def train(REFINE_TRAIN):
    
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        keep_conv = tf.placeholder(tf.float32)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        
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
                  
        # define saver
        #print (coarse_params)
        saver_coarse = tf.train.Saver(coarse_params)
        
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
            
        # fine tune 微调。。。
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print(coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
                
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(MAX_STEPS):
            index = 0
            for i in range(500): #原来是1000
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8})                
                if index % 10 == 0: #10的倍数 （0，10，20）
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 50 == 0:
                    if REFINE_TRAIN:
                        output_predict(logits_val, images_val, "data/predict_refine_%05d_%05d" % (step, i))
                    else:
                        output_predict(logits_val, images_val, "data/predict_%05d_%05d" % (step, i))
                index += 1
                
            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        coord.request_stop() #请求所有线程停止
        coord.join(threads) #等待所有的线程完成
        sess.close()
        
def train_data(NumOfImage,NumOfTrainImge):
    a=random.sample(range(NumOfImage),NumOfTrainImge)
    trains = []
    for i in range(len(a)):
        train_image_name = os.path.join("data", "hand_data", "%05d.jpg" % (a[i]))
        train_depth_name = os.path.join("data", "hand_data", "%05d.png" % (a[i]))
        trains.append((train_image_name, train_depth_name))
    
    with open('train.csv', 'w') as output:
            for (image_name, depth_name) in trains:
                output.write("%s,%s" % (image_name, depth_name))
                output.write("\n")

def main(argv=None):
    
    if not gfile.Exists(COARSE_DIR): #如果不存在 新建一个文件夹叫（COARSE_DIR）
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    
    train_data(NumOfImage,NumOfTrainImge)
    train(False)
    train(True)


if __name__ == '__main__':
    tf.app.run()
