"""
Created on 2019-03-09 2:54 PM

@author: jack.lingheng.meng
"""
"""
Created on 2019-02-27 8:36 PM

@author: jack.lingheng.meng
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli
import matplotlib.pyplot as plt

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, model_prob=0.9, model_lam=1e-2, activation=None, name="hidden"):
        self.model_prob = model_prob    # probability to keep units
        self.model_lam = model_lam      # l^2 / 2*tau
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        self.dropout_mask = self.model_bern.sample((n_in, ))

        if activation is None:
            self.activation = tf.identity
        else:
            self.activation = activation

        kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        self.model_M = tf.get_variable("{}_M".format(name), initializer=kernel_initializer([n_in, n_out])) # variational parameters
        self.model_m = tf.get_variable("{}_b".format(name), initializer=tf.zeros([n_out]))

        self.model_W = tf.matmul(tf.diag(self.dropout_mask), self.model_M)
        # self.model_W = self.model_M

    def __call__(self, X):
        output = self.activation(tf.matmul(X, self.model_W) + self.model_m)
        if self.model_M.shape[1] == 1:
            output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation,
                            kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01))
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation,
                           kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01))

def mlp_dropout(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, seed = 0, training=False):
    regularization = 0
    model_lam = 1e-2
    model_prob = 0.1

    kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)

    # Hidden Layer
    for l, h in enumerate(hidden_sizes[:-1]):
        x = tf.layers.Dropout(rate=model_prob, seed=seed)(x, training=True)
        # import pdb; pdb.set_trace()
        hidden_layer = tf.layers.Dense(units=h,
                                       activation=activation,
                                       kernel_initializer=kernel_initializer)
        x = hidden_layer(x)
        # hidden_layer.get_weights()[0]
        regularization += model_lam * (
            model_prob * tf.reduce_sum(tf.square(hidden_layer.weights[0])) +
            tf.reduce_sum(tf.square(hidden_layer.weights[1]))
        )

    # Output Layer
    x = tf.layers.Dropout(rate=model_prob, seed=seed)(x, training=True)
    output_layer = tf.layers.Dense(units=hidden_sizes[-1], activation=output_activation,
                                   kernel_initializer=kernel_initializer)
    x = output_layer(x)
    # regularization += model_lam * (
    #         model_prob * tf.reduce_sum(tf.square(output_layer.get_weights()[0])) +
    #         tf.reduce_sum(tf.square(output_layer.get_weights()[1]))
    # )

    return x, regularization

def mlp_variational(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, dropout_rate=0.1):
    # Hidden layers
    regularization = 0
    for l, h in enumerate(hidden_sizes[:-1]):
        hidden_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                        n_out=h,
                                        model_prob=1.0-dropout_rate,
                                        model_lam=1e-2,
                                        activation=activation,
                                        name="h{}".format(l+1))
        x = hidden_layer(x)
        regularization += hidden_layer.regularization
    # Output layer
    out_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                 n_out=hidden_sizes[-1],
                                 model_prob=1.0-dropout_rate,
                                 model_lam=1e-2,
                                 activation=output_activation,
                                 name="Out")
    x = out_layer(x)
    regularization += out_layer.regularization
    return x, regularization

# Created sample data.
seed=0
np.random.seed(seed)
tf.set_random_seed(seed)

n_samples = 200
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.cos(5.*X) / (np.abs(X) + 1.), 0.1).ravel()
X_pred = np.atleast_2d(np.linspace(-6., 6., num=10000)).T
y_pred = np.random.normal(np.cos(5.*X_pred) / (np.abs(X_pred) + 1.), 0.1).ravel()

X = np.hstack((X, X**2, X**3))
X_pred = np.hstack((X_pred, X_pred**2, X_pred**3))


# Create the TensorFlow model.
obs_dim = X.shape[1]
hidden_sizes = (300, 300, 300, 300)
# hidden_sizes = (100, 100)

model_X = tf.placeholder(tf.float32, [None, obs_dim])
model_X_targ = tf.placeholder(tf.float32, [None, obs_dim])

dropout_rate = 0.1#0.05

with tf.variable_scope('main'):
    q, q_reg = mlp_variational(model_X, list(hidden_sizes)+[1], tf.nn.relu, None, dropout_rate)
    # q, q_reg = mlp_dropout(model_X, list(hidden_sizes)+[1], tf.nn.relu, None)
    # q = mlp(model_X, list(hidden_sizes)+[1], tf.nn.relu, None)

with tf.variable_scope('target'):
    q_targ, q_reg_targ = mlp_variational(model_X_targ, list(hidden_sizes) + [1], tf.nn.relu, None, dropout_rate)
    # q_targ, q_reg_targ = mlp_dropout(model_X_targ, list(hidden_sizes)+[1], tf.nn.relu, None)
    # q_targ = mlp(model_X_targ, list(hidden_sizes)+[1], tf.nn.relu, None)

target_update = tf.group([tf.assign(v_targ, v_main) for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

# q = tf.squeeze(q)
# q_targ = tf.squeeze(q_targ)
# q_reg = tf.squeeze(q_reg)

# import pdb; pdb.set_trace()

model_y = tf.placeholder(tf.float32, [None])
# q_loss = tf.reduce_mean((q-model_y)**2) + q_reg / n_samples
q_loss = (tf.reduce_mean((q-model_y)**2))
model_mse = tf.reduce_mean((q-model_y)**2)

train_step = tf.train.AdamOptimizer(1e-3).minimize(q_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_summary_dir = os.path.join('./', 'tf_summary')
    writer = tf.summary.FileWriter(tf_summary_dir, sess.graph)

    training_epoches = 200#10000
    for i in range(training_epoches):
        sess.run([train_step], {model_X: X, model_y: y})
        # print('q_reg_value={}'.format(q_reg_value))
        if i % 100 == 0:
            mse = sess.run(model_mse, {model_X: X, model_y: y})
            print("Iteration {}. Mean squared error: {}.".format(i, mse))

    sess.run(target_update)
    # Sample from the posterior.
    n_post = 200#1000
    Y_post = np.zeros((n_post, X_pred.shape[0]))
    Y_post_targ = np.zeros((n_post, X_pred.shape[0]))
    for i in range(n_post):
        Y_post[i] = sess.run(q, {model_X: X_pred})
        # import pdb; pdb.set_trace()
        # Y_post_targ[i] = sess.run(q_targ, {model_X_targ: X_pred})
    # import pdb; pdb.set_trace()

# import pdb;
# pdb.set_trace()

if True:
    plt.figure(figsize=(8, 6))
    alpha = 1. / 10 #1. / 200
    for i in range(len(Y_post)):
        handle_0, = plt.plot(X_pred[:, 0], Y_post[i], "b-", alpha=alpha)
    handle_1, = plt.plot(X[:, 0], y, "r.", markersize=2)
    handle_2, = plt.plot(X_pred[:, 0], np.median(Y_post, axis=0), "g-")
    plt.title('traning_epoches={}, traning_samples={}, post_samples={} \ndropout_rate={}, hidden_sizes={}'.format(training_epoches, n_samples, n_post, dropout_rate, hidden_sizes))
    plt.ylim([-6, 6])
    plt.legend(handles=[handle_0, handle_1, handle_2],
               labels=['post sample', 'training samples', 'median of post sample'])
    plt.grid()
    plt.show()

    # plt.figure(figsize=(8, 6))
    # alpha = 1. / 10  # 1. / 200
    # for i in range(len(Y_post)):
    #     handle_0, = plt.plot(X_pred[:, 0], Y_post_targ[i], "b-", alpha=alpha)
    # handle_1, = plt.plot(X[:, 0], y, "r.", markersize=2)
    # handle_2, = plt.plot(X_pred[:, 0], np.median(Y_post_targ, axis=0), "g-")
    # plt.title('traning_epoches={}, traning_samples={}, post_samples={}'.format(training_epoches, n_samples, n_post))
    # plt.ylim([-6, 6])
    # plt.legend(handles=[handle_0, handle_1, handle_2],
    #            labels=['post sample', 'training samples', 'median of post sample'])
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # alpha = 1. / 20  # 1. / 200
    # for i in range(len(Y_post)):
    #     plt.plot(X_pred[:, 0], Y_post_targ[i], "b-", alpha=alpha)
    # plt.plot(X[:, 0], y, "r.", markersize=2)
    # plt.title('traning_epoches={}, traning_samples={}, post_samples={}'.format(training_epoches, n_samples, n_post))
    # plt.ylim([-6, 6])
    # plt.grid()
    # plt.show()