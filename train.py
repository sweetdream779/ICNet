"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""

from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf

from model import ICNet_BN
from tools import decode_labels, prepare_label, inv_preprocess
from image_reader import ImageReader

from hyperparams import *

def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use or not class weights. Values must be defined in script manually")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step = step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices

def create_loss(output, label, num_classes, ignore_label, use_w = False):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)


    #with tf.device('/cpu:0'):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred, labels = gt)

    # Make mistakes for class N more important for network
    if use_w:
        if len(CLASS_WEIGHTS) != num_classes:
            print('Incorrect class weights, it will be not used')
        else:

            mask = tf.zeros_like(loss)
            for i, w in enumerate(CLASS_WEIGHTS):
                #mask = mask + tf.cast(tf.equal(gt, i), tf.float32) * tf.constant(w)
                preds = tf.unstack(pred, axis = -1)[0]
                mask = mask + tf.cast(tf.logical_or(tf.equal(gt, i), tf.equal(preds, i)), tf.float32) * tf.constant(w)

            loss = loss * mask

    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    coord = tf.train.Coordinator()
    
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    net = ICNet_BN({'data': image_batch}, is_training=False, num_classes=args.num_classes)
    
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']

    fc_list = ['conv6_cls']

    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]
    restore_var = [v for v in tf.global_variables() if not (len([f for f in fc_list if f in v.name])) or not args.not_restore_last]
   
    for v in restore_var:
        print(v.name)

    loss_sub4 = create_loss(sub4_out, label_batch, args.num_classes, args.ignore_label, args.use_class_weights)
    loss_sub24 = create_loss(sub24_out, label_batch, args.num_classes, args.ignore_label, args.use_class_weights)
    loss_sub124 = create_loss(sub124_out, label_batch, args.num_classes, args.ignore_label, args.use_class_weights)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    
    loss = LAMBDA1 * loss_sub4 +  LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124

    reduced_loss = loss + tf.add_n(l2_losses)


    ##############################
    # visualization and summary
    ##############################


    # Processed predictions: for visualisation.

    # Sub 4
    raw_output_up4 = tf.image.resize_bilinear(sub4_out, tf.shape(image_batch)[1:3,])
    raw_output_up4 = tf.argmax(raw_output_up4, dimension = 3)
    pred4 = tf.expand_dims(raw_output_up4, dim = 3)
    # Sub 24
    raw_output_up24 = tf.image.resize_bilinear(sub24_out, tf.shape(image_batch)[1:3,])
    raw_output_up24 = tf.argmax(raw_output_up24, dimension=3)
    pred24 = tf.expand_dims(raw_output_up24, dim=3)
    # Sub 124
    raw_output_up124 = tf.image.resize_bilinear(sub124_out, tf.shape(image_batch)[1:3,])
    raw_output_up124 = tf.argmax(raw_output_up124, dimension=3)
    pred124 = tf.expand_dims(raw_output_up124, dim=3)

    images_summary = tf.py_func(inv_preprocess, [image_batch, SAVE_NUM_IMAGES, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch,SAVE_NUM_IMAGES, args.num_classes], tf.uint8)

    preds_summary4 = tf.py_func(decode_labels, [pred4, SAVE_NUM_IMAGES, args.num_classes], tf.uint8)
    preds_summary24 = tf.py_func(decode_labels, [pred24, SAVE_NUM_IMAGES, args.num_classes], tf.uint8)
    preds_summary124 = tf.py_func(decode_labels, [pred124, SAVE_NUM_IMAGES, args.num_classes], tf.uint8)
    
    total_images_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary124]), 
                                     max_outputs=SAVE_NUM_IMAGES) # Concatenate row-wise.

    total_summary = [total_images_summary]

    loss_summary = tf.summary.scalar('Total_loss', reduced_loss)

    total_summary.append(loss_summary)
    
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
    ##############################
    ##############################

    # Using Poly learning rate policy 
    if LR_SHEDULE == {}:
        base_lr = tf.constant(args.learning_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    else:
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.Variable(LR_SHEDULE.popitem()[1], tf.float32)

    lr_summary = tf.summary.scalar('Learning_rate', learning_rate)
    total_summary.append(lr_summary)
    
    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))
        
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 30)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('Restore from pre-trained model...')
        net.load(args.restore_from, sess, ignore_layers = fc_list)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    summ_op = tf.summary.merge(total_summary)
    
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        if LR_SHEDULE != {}:
            if step >= LR_SHEDULE.keys()[0]:
                tf.assign(learning_rate, LR_SHEDULE.popitem()[0])

        feed_dict = {step_ph: step}
        if step % args.save_pred_every == 0:
            
            loss_value, loss1, loss2, loss3, _, summary =\
                sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op, summ_op], feed_dict = feed_dict)

            save(saver, sess, args.snapshot_dir, step)
            summary_writer.add_summary(summary, step)

        else:
            loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)
            
        duration = time.time() - start_time
        #print('shape', sess.run(tf.shape(sub124_out)))
        #quit()
        print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(step, loss_value, loss1, loss2, loss3, duration))
        
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
