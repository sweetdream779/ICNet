r"""Saves out a GraphDef containing the architecture of the model."""

from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.platform import gfile

from model import ICNet_BN
from tools import decode_labels, prepare_label, inv_preprocess
from image_reader import ImageReader
from inference import preprocess, check_input

from hyperparams import *

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

#tf.app.flags.DEFINE_integer(
#    'image_size', None,
#    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), 3)

        x = tf.placeholder(name = 'input', dtype = tf.float32, shape = (None, shape[0], shape[1], 3))

        img_tf = tf.cast(x, dtype=tf.float32)
        # Extract mean.
        img_tf -= IMG_MEAN

        print(img_tf)
        # Create network.
        net = ICNet_BN({'data': img_tf}, is_training = False, num_classes = NUM_CLASSES)

        raw_output = net.layers['conv6_cls']
        output = tf.image.resize_bilinear(raw_output, tf.shape(img_tf)[1:3,], name = 'raw_output')
        output = tf.argmax(output, dimension = 3)
        pred = tf.expand_dims(output, dim = 3, name = 'indices')

        # Adding additional params to graph. It is necessary also to point them as outputs in graph freeze conversation, otherwise they will be cuted
        tf.constant(label_colours, name = 'label_colours')
        tf.constant(label_names, name = "label_names")
        
        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), 3)
        tf.constant(shape, name = 'input_size')
        tf.constant(["indices"], name = "output_name")

        graph_def = graph.as_graph_def()
        with gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
            print('Successfull written to', FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
