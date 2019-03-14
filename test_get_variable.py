"""
Created on 2019-02-27 10:32 PM

@author: jack.lingheng.meng
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.distributions import Bernoulli


class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, model_prob=0.9, model_lam=1e-2, name="hidden"):
        self.model_prob = model_prob    # probability to keep units
        self.model_lam = model_lam      # l^2 / 2*tau
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        # with tf.variable_scope("variational_dense"):
        self.model_M = tf.get_variable("{}_M".format(name), initializer=tf.truncated_normal([n_in, n_out], stddev=0.01))
        self.model_m = tf.get_variable("{}_b".format(name), initializer=tf.zeros([n_out]))
        self.model_W = tf.matmul(
            tf.diag(self.model_bern.sample((n_in, ))), self.model_M
        )

    def __call__(self, X, activation=tf.identity):
        if activation is None:
            activation = tf.identity
        output = activation(tf.matmul(X, self.model_W) + self.model_m)
        # if self.model_M.shape[1] == 1:
        #     output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )

def mlp_variational(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    # Hidden layers
    regularization = 0  # tf.placeholder(dtype=tf.float32, shape=(1,))
    for l, h in enumerate(hidden_sizes[:-1]):
        hidden_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                        n_out=h,
                                        model_prob=0.9,
                                        model_lam=1e-2,
                                        name="h{}".format(l + 1))
        x = hidden_layer(x, activation)
        regularization += hidden_layer.regularization
    # Output layer
    out_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                 n_out=hidden_sizes[-1],
                                 model_prob=0.9,
                                 model_lam=1e-2,
                                 name="Out")
    x = out_layer(x, output_activation)
    regularization += out_layer.regularization
    return x, regularization

# Created sample data.
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


model_L = VariationalDense(n_feats, n_hidden, model_prob, model_lam, 'h1')
model_out = model_L(model_X, tf.nn.relu)
model_L = VariationalDense(n_hidden, n_hidden, model_prob, model_lam, 'h2')
model_out = model_L(model_out, tf.nn.relu)
model_L = VariationalDense(n_hidden, 1, model_prob, model_lam, 'output')
model_pred = model_L(model_out)
model_pred = tf.squeeze(model_pred, axis=1)

model_sse = tf.reduce_sum(tf.square(model_y - model_pred))
model_mse = model_sse / n_samples
model_loss = (
    # Negative log-likelihood.
    model_sse
    # +
    # # Regularization.
    # model_L_1.regularization +
    # model_L_2.regularization +
    # model_L_3.regularization
) / n_samples

train_step = tf.train.AdamOptimizer(1e-3).minimize(model_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train_step, {model_X: X, model_y: y})
        if i % 100 == 0:
            mse = sess.run(model_mse, {model_X: X, model_y: y})
            print("Iteration {}. Mean squared error: {:.4f}.".format(i, mse))

    # Sample from the posterior.
    n_post = 1000
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
