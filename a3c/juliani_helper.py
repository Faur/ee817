from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
# import random
import tensorflow as tf
import scipy.signal
# import matplotlib.pyplot as plt
# import scipy.misc
# import os
# import csv
# import itertools
# import tensorflow.contrib.slim as slim

### "Helper Functions"
def update_target_graph(from_scope, to_scope):
    """ Op that copies one set of variables to another
        Use to set local network parameters to those of global network. """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(frame):
    return frame

def discount(x, gamma):
    # TODO: Manually implement this
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalized_columns_initializer(std=1.0):
    """Initialize weights for policy and value output layers"""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.abs(out).sum(axis=0, keepdims=True)
        return tf.constant(out)
    return _initializer


### "helper.py"
def processState(s):
    return s

def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
  """ This code allows gifs to be saved of the training episode for use in the Control Center."""
  import moviepy.editor as mpy
  
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  
  def make_mask(t):
    try:
      x = salIMGS[int(len(salIMGS)/duration*t)]
    except:
      x = salIMGS[-1]
    return x

  clip = mpy.VideoClip(make_frame, duration=duration)
  if salience == True:
    mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
    clipB = clip.set_mask(mask)
    clipB = clip.set_opacity(0)
    mask = mask.set_opacity(0.1)
    mask.write_gif(fname, fps = len(images) / duration,verbose=False)
    #clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
  else:
    clip.write_gif(fname, fps = len(images) / duration,verbose=False)

# def updateTarget(op_holder, sess):


