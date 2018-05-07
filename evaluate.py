from __future__ import print_function
import argparse
import os
import sys
import time
import shutil
import zipfile

from PIL import Image
import tensorflow as tf
import numpy as np

from model import ICNet_BN
from tools import decode_labels
from image_reader import ImageReader
import logging

from train import IMG_MEAN, NUM_CLASSES, INPUT_SIZE, IGNORE_LABEL

def calc_size(filename):
    size = 0
    with open(filename, 'r') as f:
        for line in f:
            size = size + 1

    return size

SAVE_DIR = './output/'

DATA_LIST_PATH = '/home/irina/Desktop/dataset_person/val.txt'

snapshot_dir = './snapshots'
best_models_dir = './best_models'

num_classes = NUM_CLASSES

num_steps = calc_size(DATA_LIST_PATH) # numbers of images in validation set
time_list = []
INTERVAL = 120
INPUT_SIZE = INPUT_SIZE.split(',')
INPUT_SIZE = [int(INPUT_SIZE[0]), int(INPUT_SIZE[1])]
IGNORE_LABEL = IGNORE_LABEL
batch_size = 1


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Text file with pairs image-answer")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--snapshot-dir", type=str, default=snapshot_dir,
                        help="Path to load")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--repeated-eval", action="store_true",
                        help="Run repeated evaluation for every checkpoint.")
    parser.add_argument("--ignore-zero", action="store_true",
                        help="If true, zero class will be ignored for total score")
    parser.add_argument("--best-models-dir", type=str, default=best_models_dir,
                        help="If set, best mIOU checkpoint will be saved in that dir in .zip format")
    parser.add_argument("--eval-interval", type=int, default=INTERVAL,
                        help="How often to evaluate model, seconds")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Size of batch")



    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def save_model(step, iou, checkpint_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    files = os.listdir(checkpint_dir)

    files = [os.path.abspath(checkpint_dir + '/' + f) for f in files]

    filename = list([f for f in files if str(step) in f])

    if len(filename) != 3 and len(filename) != 4:
        return

    iou = '{0:.4f}'.format(iou)
    zipfile_name = output_dir + '/miou_{0}.zip'.format(iou)

    print('Saving stpe {} with mIOU {} in file {}'.format(step, iou, zipfile_name))

    zf = zipfile.ZipFile(zipfile_name, "w", zipfile.ZIP_DEFLATED)
    for f in filename:
        zf.write(f, os.path.basename(f))
    zf.close()

def load_last_best_iou(dir):

    if not os.path.exists(dir):
        return 0.0

    files = os.listdir(dir)

    best_iou = 0.0
    for f in files:
        
        iou = float(f[f.rfind('miou_') + 5 : f.rfind('.')])
        if iou > best_iou:
            best_iou = iou
    
    return best_iou

def evaluate_checkpoint(model_path, args):
    coord = tf.train.Coordinator()

    tf.reset_default_graph()

    reader = ImageReader(
            args.data_list,
            INPUT_SIZE,
            random_scale = False,
            random_mirror = False,
            ignore_label = IGNORE_LABEL,
            img_mean = IMG_MEAN,
            coord = coord,
            train = False)
    image_batch, label_batch = reader.dequeue(args.batch_size)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    # Create network.
    net = ICNet_BN({'data': image_batch}, is_training = False, num_classes = num_classes)
    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['conv6_cls']


    raw_output_up = tf.image.resize_bilinear(raw_output, size = INPUT_SIZE, align_corners = True)
    raw_output_up = tf.argmax(raw_output_up, dimension = 3)
    pred = tf.expand_dims(raw_output_up, dim = 3)

    # mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(label_batch, [-1,])
    if args.ignore_zero:
        indices = tf.squeeze(tf.where(
            tf.logical_and(
                tf.less_equal(raw_gt, num_classes - 1),
                tf.greater(raw_gt, 0)
                ),), 
            1)
    else:
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)

    #indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)

    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    metric, op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes = num_classes)

    mIoU, update_op = metric, op
    
    # Summaries
    miou_op = tf.summary.scalar('mIOU', mIoU)
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                        time.gmtime()))
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(var_list = restore_var)
    load(saver, sess, model_path)
    

    for step in range(num_steps):
        preds, _ = sess.run([pred, update_op])

        if step % 500 == 0:
            print('Finish {0}/{1}'.format(step + 1, num_steps))

    iou, summ = sess.run([mIoU, miou_op])

    sess.close()

    coord.request_stop()
    #coord.join(threads)

    return summ, iou

#########################################################


def main():
    args = get_arguments()

    if args.repeated_eval:

        last_evaluated_model_path = None

        while True:
            start = time.time()
            
            best_iou = load_last_best_iou(args.best_models_dir)
            
            model_path = tf.train.latest_checkpoint(args.snapshot_dir)

            if not model_path:
                logging.info('No model found')
            elif model_path == last_evaluated_model_path:
                logging.info('Found already evaluated checkpoint. Will try again in %d '
                    'seconds', args.eval_interval)
            else:
                global_step = int(os.path.basename(model_path).split('-')[1])
                last_evaluated_model_path = model_path
                number_of_evaluations = 0

                eval_path = args.snapshot_dir + '/eval'
                if not (os.path.exists(eval_path)):
                    os.mkdir(eval_path)
                
                summary_writer = tf.summary.FileWriter(eval_path)

                summ, iou = evaluate_checkpoint(last_evaluated_model_path, args)
                print('Step', global_step, ', mIOU:', iou)

                if iou > best_iou:
                    if len(args.best_models_dir):
                        save_model(global_step, iou, args.snapshot_dir, args.best_models_dir)
                    best_iou = iou

                print('Best for now mIOU: {}'.format(best_iou))

                summary_writer.add_summary(summ, global_step)
                number_of_evaluations += 1
            
                ########################

                time_to_next_eval = start + args.eval_interval - time.time()

                if time_to_next_eval > 0:
                    
                    time.sleep(time_to_next_eval)

    # run once. Not tested yet
    else:
        
        model_path = tf.train.latest_checkpoint(args.snapshot_dir)
        global_step = int(os.path.basename(model_path).split('-')[1])
        summ, iou = evaluate_checkpoint(model_path, args)
        print('Step', global_step, ', mIOU:', iou)


if __name__ == '__main__':
    main()
