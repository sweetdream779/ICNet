from __future__ import print_function

import sys
from pathlib import Path
sys.path.append('../')
sys.path.append('./')
sys.path.append('./datasets')

import argparse
import os
import copy

import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from model import ICNet_BN
from tools import decode_labels

import image_reader

import train
from train import INPUT_SIZE, IMG_MEAN, NUM_CLASSES

import cv2
'''
INPUT_SIZE = '2049,1025'
NUM_CLASSES = 19
'''

def GetAllFilesListRecusive(path, extensions):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all
num_classes = NUM_CLASSES

snapshot_dir = './snapshots/'

SAVE_DIR = './output/'

def calculate_perfomance(sess, input, raw_output, shape, runs = 1000, batch_size = 1):

    start = time.time()

    print('Calculating inference time...\n')
    # To exclude numpy generating time
    N = 10
    for i in range(0, N):
        img = np.random.random((batch_size, shape[0], shape[1], 3))
    stop = time.time()
    
    # warm up
    sess.run(raw_output, feed_dict = {input : img})

    time_for_generate = (stop - start) / N

    start = time.time()
    for i in range(runs):
        img = np.random.random((batch_size, shape[0], shape[1], 3))
        sess.run(raw_output, feed_dict = {input : img})

    stop = time.time()

    inf_time = ((stop - start) / float(runs)) - time_for_generate

    print('Average inference time: {}'.format(inf_time))


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.",
                        required=True)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--snapshots-dir", type=str, default=snapshot_dir,
                        help="Path to checkpoints.")
    parser.add_argument("--pb-file", type=str, default='',
                        help="Path to to pb file, alternative for checkpoint. If set, checkpoints will be ignored")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="If true, will output weighted images")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Size of batch for time measure")
    parser.add_argument("--measure-time", action="store_true", default=False,
                        help="Evaluate only model inference time")
    parser.add_argument("--runs", type=int, default=100,
                        help="Repeats for time measure. More runs - longer testing - more precise results")


    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    img = cv2.imread(img_path)
    
    shape = INPUT_SIZE.split(',')
    '''
    h = img.shape[0]
    w = img.shape[1]
    blank_img = np.zeros((800,800,3),np.uint8)
    if(h>800 or  w>800):
        if(w > h):
            new_w = 800
            scale = w/new_w
            new_h = h/scale
            
        else:
            new_h = 800
            scale = h / new_h
            new_w = w/scale

        img = cv2.resize(img, (int(new_w), int(new_h)))
        blank_img[0:int(new_h), 0:int(new_w)] = img
    else:
        blank_img[0:h, 0:w] = img

    img = blank_img
    cv2.imshow("df", img)
    '''

    img = cv2.resize(img, (int(shape[0]), int(shape[1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('input image shape: ', img.shape)

    return img, filename

def preprocess(img):
    # Convert RGB to BGR
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img

    img = tf.cast(img, dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    
    img = tf.expand_dims(img, dim=0)

    return img

def check_input(img):
    
    ori_h, ori_w = img.get_shape().as_list()[1:3]

    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h/32) + 1) * 32
        new_w = (int(ori_w/32) + 1) * 32
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
        
        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]
    

    return img, shape

def load_from_checkpoint(shape, path):
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    img_tf = preprocess(x)

    img_tf, n_shape = check_input(img_tf)

    # Create network.
    net = ICNet_BN({'data': img_tf}, is_training = False, num_classes = num_classes)

    # Predictions.
    raw_output = net.layers['conv6_cls']
    print('raw_output', raw_output)
    output = tf.image.resize_bilinear(raw_output, tf.shape(img_tf)[1:3,])
    output = tf.argmax(output, dimension = 3)
    pred = tf.expand_dims(output, dim = 3)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()
    
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    
    #net.load('./model/icnet_cityscapes_trainval_90k_bnnomerge.npy', sess)
    return sess, pred, x

def load_from_pb(shape, path):
    segment_graph = tf.Graph()
    with segment_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)

            tf.import_graph_def(seg_graph_def, name = '')

            x = segment_graph.get_tensor_by_name('input:0')

            pred = segment_graph.get_tensor_by_name('indices:0')

            config = tf.ConfigProto()
            config.graph_options.optimizer_options.do_common_subexpression_elimination = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            config.allow_soft_placement = True
            config.log_device_placement = False

            sess = tf.Session(graph = segment_graph, config = config)

    return sess, pred, x

def main():
    args = get_arguments()
    
    if args.img_path[-4] != '.':
        files = GetAllFilesListRecusive(args.img_path, ['.jpg', '.jpeg', '.png'])
    else:
        files = [args.img_path]


    shape = INPUT_SIZE.split(',')
    shape = (int(shape[0]), int(shape[1]), 3)

    if args.pb_file == '':
        sess, pred, x = load_from_checkpoint(shape, args.snapshots_dir)
    else:
        sess, pred, x = load_from_pb(shape, args.pb_file)

    if args.measure_time:
        calculate_perfomance(sess, x, pred, shape, args.runs, args.batch_size)
        quit()

    for path in files:

        img, filename = load_img(path)
        
        orig_img = copy.deepcopy(img)

        if args.pb_file != '':
            img = np.expand_dims(img, axis = 0)

        t = time.time()
        preds = sess.run(pred, feed_dict = {x: img})

        #print('time: ', time.time() - t)
        #print('output shape: ', preds.shape)

        msk = decode_labels(preds, num_classes=num_classes)
        im = msk[0]
        #print('im', im.shape)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        if args.weighted:
            indx = (im == [0, 0, 0])
            print(im.shape, img.shape)
            im = cv2.addWeighted(im, 0.7, img, 0.7, -15)
            im[indx] = img[indx]

        cv2.imwrite(args.save_dir + filename.replace('.jpg', '.png'), im)
        cv2.imshow("sdf",im)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
