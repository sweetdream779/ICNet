import os

import numpy as np
import tensorflow as tf

from hyperparams import *

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval = 0.5, maxval = 1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
   
    return img, label

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]

    #size = tf.stack([crop_h, crop_w, tf.constant([4])], axis=0)
    #size = tf.squeeze(size)

    #combined_crop = tf.random_crop(combined, size)
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))

    return img_crop, label_crop  

def read_labeled_image_list(data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(image)
        masks.append(mask)
        
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean, train = True): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    '''
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    
    # Extract mean.
    image = img - img_mean

    answer = tf.image.decode_png(label_contents, channels=1)
    
    if input_size is None:
        print('WTF?!?!')
        quit()

    h, w = input_size

    # Randomly scale the images and labels.
    # if random_scale:
    #     img, label = image_scaling(img, label)

    # Randomly mirror the images and labels.
    if random_mirror:
        image, answer = image_mirroring(image, answer)

    if train:
        
        def crop_img(a, b, img = image, label = answer):
            # Randomly crops the images and labels.
            h_rate = tf.random_uniform(shape = [1], minval = 0.4, maxval = 1.0, dtype = tf.float32)
            w_rate = tf.random_uniform(shape = [1], minval = 0.4, maxval = 1.0, dtype = tf.float32)
            random_h = tf.cast(tf.cast(h, tf.float32) * h_rate, tf.int32)
            random_w = tf.cast(tf.cast(w, tf.float32) * w_rate, tf.int32)

            #img, label = random_crop_and_pad_image_and_labels(img, label, random_h, random_w, ignore_label)
            img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

            return img, label

        
        img, label = crop_img(0, 0)

        if CROP_MUSTHAVE_CLASS_INDEX != -1:

            def is_img_not_ok(img = img, label = label): 
                mask = tf.equal(label, CROP_MUSTHAVE_CLASS_INDEX)
                any = tf.reduce_any(mask)
                return tf.logical_not(any)
            
            img, label = tf.while_loop(is_img_not_ok, crop_img, loop_vars=[img, label]) 

    else:
        img = image
        label = answer

            

    img = tf.image.resize_images(img, input_size)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), input_size)
    label = tf.squeeze(label, squeeze_dims=[0])

    img.set_shape([h, w, 3])
    label.set_shape([h, w, 1])

    return img, label
    '''
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        if random_scale:
            img, label = image_scaling(img, label)

        if random_mirror:
            img, label = image_mirroring(img, label)

        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label
    

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_list, input_size, 
                 random_scale, random_mirror, ignore_label, img_mean, coord, train = True):
        '''Initialise an ImageReader.
        
        Args:
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.train = train
        
        self.image_list, self.label_list = read_labeled_image_list(self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, ignore_label, img_mean, self.train) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch



def random_crop_and_pad_image(image, crop_h, crop_w):

    image_shape = tf.shape(image)

    combined_pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]

    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    
    img_crop.set_shape((crop_h, crop_w, 3))

    return img_crop


def read_image_from_disk(path, input_size, img_mean, train = True): # optional pre-processing arguments

    filename = img_path.split('/')[-1]
    img_contents = tf.read_file(path)

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    if input_size is not None:
        h, w = input_size

        img, label = random_crop_and_pad_image(img, h, w)

    return img, filename