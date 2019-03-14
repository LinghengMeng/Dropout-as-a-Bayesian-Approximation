"""
Created on 2019-02-27 9:24 PM

@author: jack.lingheng.meng
"""
# Created sample data.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_samples = 2000
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.cos(5.*X) / (np.abs(X) + 1.), 0.1).ravel()
X_pred = np.atleast_2d(np.linspace(-3., 3., num=100)).T
X = np.hstack((X, X**2, X**3))
X_pred = np.hstack((X_pred, X_pred**2, X_pred**3))

# Create the TensorFlow model.
n_feats = X.shape[1]
n_hidden = 100
model_prob = 0.9
model_lam = 1e-2
model_X = tf.placeholder(tf.float32, [None, n_feats])
model_y = tf.placeholder(tf.float32, [None])

model_out_1 = tf.layers.dense(model_X, n_hidden, tf.nn.relu)
model_out_2 = tf.layers.dense(model_out_1, n_hidden, tf.nn.relu)
model_pred = tf.layers.dense(model_out_2, 1)

model_sse = tf.reduce_sum(tf.square(model_y - model_pred))
model_mse = model_sse / n_samples
model_loss = model_mse
train_step = tf.train.AdamOptimizer(1e-3).minimize(model_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train_step, {model_X: X, model_y: y})
        if i % 100 == 0:
            mse = sess.run(model_mse, {model_X: X, model_y: y})
            print("Iteration {}. Mean squared error: {:.4f}.".format(i, mse))

    # Sample from the posterior.
    n_post = 10000
    Y_post = np.zeros((n_post, X_pred.shape[0]))
    for i in range(n_post):
        Y_post[i] = sess.run(model_pred, {model_X: X_pred})

if True:
    plt.figure(figsize=(8, 6))
    for i in range(n_post):
        plt.plot(X_pred[:, 0], Y_post[i], "b-", alpha=1. / 200)
    # plt.plot(X[:, 0], y, "r.")
    plt.title('n_samples={}, n_post={}'.format(n_samples, n_post))
    plt.grid()
    plt.show()