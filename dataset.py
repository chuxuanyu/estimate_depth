import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import csv
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_inputs(self, csv_file_path):
        
        
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        
        feature = tf.stack([filename, depth_filename])
        
        # input
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)    #转换数据类型   
        
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0]) #使其为0-1
        #depth = tf.cast(depth, tf.int64)
        
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth) #输出为-1，0，1 根据符号
       
        # generate batch
        images, depths, invalid_depths,features = tf.train.batch(
            [image, depth, invalid_depth,feature],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 *self.batch_size,#????最多队列里面有多少人
        )
        
        return images, depths, invalid_depths,features      
        
def output_predict(depths, images, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)

def output_save(depths1, depths2, images,step,output_dir):
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image,depth1, depth2) in enumerate(zip(images,depths1, depths2)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_d_image.png" % (output_dir, step)
        pilimg.save(image_name)
        depth1 = depth1.transpose(2, 0, 1)
        depth2 = depth2.transpose(2, 0, 1)
        if np.max(depth1) != 0:
            ra_depth1 = (depth1/np.max(depth1))*255.0
        else:
            ra_depth1 = depth1*255.0
        if np.max(depth2) != 0:
            ra_depth2 = (depth2/np.max(depth2))*255.0
        else:
            ra_depth2 = depth2*255.0
        depth1_pil = Image.fromarray(np.uint8(ra_depth1[0]), mode="L")
        depth2_pil = Image.fromarray(np.uint8(ra_depth2[0]), mode="L")
        depth1_name = "%s/%05d_a_coarse.png" % (output_dir, step)
        depth2_name = "%s/%05d_b_refine.png" % (output_dir, step)
        depth1_pil.save(depth1_name)
        depth2_pil.save(depth2_name)