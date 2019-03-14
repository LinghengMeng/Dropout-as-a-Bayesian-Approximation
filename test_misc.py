import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli
import matplotlib.pyplot as plt

# def dropout_masks_placeholders(x, hidden_sizes=(32,), model_prob=0.9):
#     model_bern = Bernoulli(probs=model_prob, dtype=tf.float32)
#     dropout_masks_ph = []
#     for l, size in enumerate(( x.shape.as_list()[1], *hidden_sizes)):
#         dropout_masks_ph.append(model_bern.sample((size, )))
#     return dropout_masks_ph
#
# obs_dim = 12
# x = tf.placeholder(tf.float32, [None, obs_dim])
# hidden_sizes = (300, 300, 300, 300)
# model_prob=0.9
#
# dropout_masks_ph = dropout_masks_placeholders(x, hidden_sizes, model_prob)
#
# with tf.Session() as sess:
#     dropout_masks = sess.run(dropout_masks_ph, feed_dict={x: np.random.normal(size=(100, obs_dim))})
#     import pdb;
#
#     pdb.set_trace()
#     print(dropout_masks)

def generate_dropout_mask_placeholders(x_dim, hidden_sizes=(32,)):
    dropout_mask_placeholders = []
    for l, size in enumerate((x_dim, *hidden_sizes)):
        dropout_mask_placeholders.append(tf.placeholder(dtype=tf.float32, name='dropout_mask_{}'.format(l)))
    return dropout_mask_placeholders

def update_dropout_masks(x_dim, hidden_sizes=(32,), model_prob=0.9):
    model_bern = Bernoulli(probs=model_prob, dtype=tf.float32)
    new_dropout_masks = []
    for l, size in enumerate((x_dim, *hidden_sizes)):
        new_dropout_masks.append(model_bern.sample((size,)))
    return new_dropout_masks

# Create the TensorFlow model.
obs_dim = 12
hidden_sizes = (300, 300, 300, 300)
# hidden_sizes = (100, 100)

model_X = tf.placeholder(tf.float32, [None, obs_dim])
model_X_targ = tf.placeholder(tf.float32, [None, obs_dim])

dropout_rate = 0.1

new_dropout_masks = update_dropout_masks(obs_dim, hidden_sizes, model_prob=1.0-dropout_rate)
dropout_mask_phs = generate_dropout_mask_placeholders(obs_dim, hidden_sizes)

with tf.Session() as sess:
    for i in range(5):
        dropout_masks = sess.run(new_dropout_masks)
        print(dropout_masks[0])