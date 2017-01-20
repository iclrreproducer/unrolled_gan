import os
from os.path import expanduser
import numpy as np
import tensorflow as tf
from scipy.misc import imsave

from utils import SMORMS3, Momentum, SessionWrap, random_batch, dense

def lrelu(x, leakiness=0.2):
    return tf.maximum(leakiness * x, x)

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def save_images(fname, flat_img, width=28, height=28, sep=3):
    N = flat_img.shape[0]
    pdim = int(np.ceil(np.sqrt(N)))
    image = np.zeros((pdim * (width+sep), pdim * (height+sep)))
    for i in range(N):
        row = int(i / pdim) * (height+sep)
        col = (i % pdim) * (width+sep)
        image[row:row+width, col:col+height] = flat_img[i].reshape(width, height)
    imsave(fname, image)

def load_mnist(dir_name="mnist"):
    home = expanduser("~")
    data_dir = os.path.join(home, "data", dir_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
        
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i,int(y[i])] = 1.0
        
    return X/255.,y_vec

def generator(z, n_hid=500, isize=28*28, reuse=False, use_bn=False):
    bn1 = batch_norm(name='bn1')
    bn2 = batch_norm(name='bn2')
    hid = dense(z, n_hid, scope='l1', reuse=reuse)
    if use_bn:
        hid = tf.nn.relu(bn1(hid, train=True))
    else:
        hid = tf.nn.relu(hid)
    hid = dense(hid, n_hid, scope='l2', reuse=reuse)
    if use_bn:
        hid = tf.nn.relu(bn2(hid, train=True))
    else:
        hid = tf.nn.relu(hid)
    out = tf.nn.sigmoid(dense(hid, isize, scope='g_out', reuse=reuse))
    return out

def discriminator(x, z_size, n_hid=500, isize=28*28, reuse=False):
    #bn1 = batch_norm(name='bn1')
    #bn2 = batch_norm(name='bn2')
    hid = dense(x, n_hid, scope='l1', reuse=reuse, normalized=True)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    hid = dense(hid, n_hid, scope='l2', reuse=reuse, normalized=True)
    #hid = tf.nn.dropout(hid, 0.2)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    out = dense(hid, 1, scope='d_out', reuse=reuse)
    return out

def discriminator_from_params(x, params, isize=28*28, n_hid=100):
    #bn1 = batch_norm(name='bn1')
    #bn2 = batch_norm(name='bn2')
    hid = dense(x, n_hid, scope='l1', params=params[:2], normalized=True)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    hid = dense(hid, n_hid, scope='l2', params=params[2:4], normalized=True)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    out = dense(hid, 1, scope='d_out', params=params[4:])
    return out

def train(loss_d, loss_g, opt_d, opt_g, data, feed_x, feed_z, z_gen, n_steps, batch_size,
          d_steps=1, g_steps=1, d_pretrain_steps=1,
          session=None, callbacks=[]):
    with SessionWrap(session) as sess:

        sess.run(tf.initialize_all_variables())
        for t in range(n_steps):
            for i in range(d_steps):
                x = random_batch(data, size=batch_size)
                z = z_gen()
                _,curr_loss_d = sess.run([opt_d, loss_d], feed_dict = { feed_x : x, feed_z : z})
            if t > d_pretrain_steps:
                for i in range(g_steps):
                    z = z_gen()
                    _,curr_loss_g = sess.run([opt_g, loss_g], feed_dict = { feed_z : z, feed_x : x})
            else:
                curr_loss_g = 0.    
            for callback in callbacks:
                callback(t, curr_loss_d, curr_loss_g)

g = tf.Graph()
x_data, y_data = load_mnist()

x_flat = x_data.reshape((-1, 28*28))

###### NOTE #######
# set number of unrolling steps and whether to use batch norm here

# This for example does not work, this also does not work when using a different optimizer (see SMORMS3 below)
lookahead = 5
use_bn = False
# but this does work (as does any setup with use_bn True)
#lookahead = 1
#use_bn = True
g_lr = 0.005
d_lr = 0.0001

eps = 1e-6
batch_size = 128
z_size = 100
g_steps = 1
d_steps = 1
steps = 100000
d_pretrain_steps = 1
isize = 28*28
with g.as_default():
    x_tf = tf.placeholder(tf.float32, shape=(batch_size, isize))
    z_tf = tf.placeholder(tf.float32, shape=(batch_size, z_size))
    with tf.variable_scope('G') as scope:
        x_gen = generator(z_tf, use_bn=use_bn)
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        g_prior = 0.
        for param in g_params:
            g_prior += 0. * tf.reduce_sum(tf.square(param))

    with tf.variable_scope('D') as scope:
        disc_out = discriminator(tf.concat_v2([x_tf,x_gen], 0), z_size)
        disc_real = disc_out[:batch_size, :]
        disc_fake = disc_out[batch_size:, :]
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat_v2([disc_real, disc_fake], 0),
                                                                    labels=tf.concat_v2([tf.ones_like(disc_real),
                                                                                tf.zeros_like(disc_fake)], 0)))


    # select optimizer for d
    #optimizer_d = Momentum(learning_rate=1e-3, mdecay=0.5)
    optimizer_d = SMORMS3(learning_rate=d_lr)
    opt_d = optimizer_d.minimize(loss_d, var_list=d_params)

    # unroll optimizer for G
    opt_vars = None
    next_d_params = d_params
    if lookahead > 0:
        for i in range(lookahead):
            disc_out_g = discriminator_from_params(tf.concat_v2([x_tf, x_gen], 0), next_d_params)
            disc_real_g = disc_out_g[:batch_size, :]
            disc_fake_g = disc_out_g[batch_size:, :]
            loss_d_tmp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat_v2([disc_real_g, disc_fake_g], 0),
                                                                                labels=tf.concat_v2([tf.ones_like(disc_real_g),
                                                                                          tf.zeros_like(disc_fake_g)], 0)))

            grads = tf.gradients(loss_d_tmp, next_d_params)
            next_d_params, opt_vars = optimizer_d.unroll_step(grads, next_d_params, opt_vars=opt_vars)
    else:
        disc_out_g = discriminator_from_params(tf.concat_v2([x_tf, x_gen], 0), next_d_params)
        disc_fake_g = disc_out_g[batch_size:, :]
    #disc_out_g = discriminator_from_params(x_gen, next_d_params)
    #disc_fake_g = disc_out_g[batch_size:, :]
    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g, labels=tf.ones_like(disc_fake_g)))

    
    loss_generator = loss_g
    #optimizer_g = Momentum(learning_rate=1e-2, mdecay=0.5)
    optimizer_g = SMORMS3(learning_rate=g_lr)

    opt_g = optimizer_g.minimize(loss_generator, var_list=g_params)

    session = tf.Session()

    def z_gen():
        return np.random.uniform(-1, 1, size=(batch_size, z_size))

    z_vis = z_gen()
    def logging(t, curr_loss_d, curr_loss_g):        
        if t % 100 == 0:
            print("{} loss D = {} loss G = {}".format(t, curr_loss_d, curr_loss_g))
        if t % 500 == 0:
            save_images('samples.png', session.run(x_gen, feed_dict={z_tf : z_vis}))

    train(loss_d, loss_generator, opt_d, opt_g, x_flat, x_tf, z_tf, z_gen,
          steps, batch_size,
          g_steps=g_steps, d_steps=d_steps, d_pretrain_steps=d_pretrain_steps,
          session=session,
          callbacks=[logging])
