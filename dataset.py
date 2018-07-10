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
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 *self.batch_size,#????最多队列里面有多少人
        )
        
        return images, depths, invalid_depths
'''
Obtain the test data
#'''
def get_test_data(csv_file_path):
    test_n = sum(1 for line in open(csv_file_path))
    images = np.zeros([test_n, IMAGE_HEIGHT, IMAGE_WIDTH, 3],dtype=np.float32)
    depths = np.zeros([test_n, TARGET_HEIGHT, TARGET_WIDTH],dtype=np.float32)
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for i,j in enumerate(csv_reader):
            img_path, depth_path = j
            image = Image.open(img_path)
            image = image.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
            image = np.asarray(image, dtype=np.float32)
            images[i,:,:,:] = image
            depth = Image.open(depth_path)
            depth = depth.resize([TARGET_WIDTH, TARGET_HEIGHT])
            depth = np.asarray(depth, dtype=np.float32)/255.0
            depths[i,:,:] = depth
    invalid_depths = np.sign(depths)
    return images, depths, invalid_depths
            
        
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
