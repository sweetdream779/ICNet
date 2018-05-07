import scipy.io as sio
import numpy as np
from PIL import Image
import tensorflow as tf

from hyperparams import *
'''
label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                 # 0 = road, 1 = sidewalk, 2 = building
                 ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                 # 3 = wall, 4 = fence, 5 = pole
                 ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                 # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                 ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                 # 9 = terrain, 10 = sky, 11 = person
                 ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                 # 12 = rider, 13 = car, 14 = truck
                 ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                 # 15 = bus, 16 = train, 17 = motocycle
                 ,(119, 10, 32), (1, 1, 1),  (222, 101, 215)

                 ,(20, 60, 80), (220, 79, 89), (20, 20, 230)

                 ,(60, 60, 110), (0, 100, 50), (10, 100, 230)

                 ,(44, 88, 166)]
#                 # 18 = bicycle, 19 = void label
'''
def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape

    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          
          for k_, k in enumerate(j):

              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs