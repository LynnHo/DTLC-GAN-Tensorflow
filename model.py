from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl


# ==============================================================================
# =                                    alias                                   =
# ==============================================================================

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
# batch_norm = partial(slim.batch_norm, scale=True)
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
layer_norm = slim.layer_norm
instance_norm = slim.instance_norm


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

def _get_norm_fn(norm_name, is_training):
    if norm_name == 'none':
        norm = None
    elif norm_name == 'batch_norm':
        norm = partial(batch_norm, is_training=is_training)
    elif norm_name == 'instance_norm':
        norm = instance_norm
    elif norm_name == 'layer_norm':
        norm = layer_norm
    return norm


def G(z, c, dim=64, is_training=True):
    norm = _get_norm_fn('batch_norm', is_training)
    fc_norm_relu = partial(fc, normalizer_fn=norm, activation_fn=relu)
    dconv_norm_relu = partial(dconv, normalizer_fn=norm, activation_fn=relu)

    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        y = tf.concat([z, c], axis=1)
        y = fc_norm_relu(y, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_norm_relu(y, dim * 4, 4, 2)
        y = dconv_norm_relu(y, dim * 2, 4, 2)
        y = dconv_norm_relu(y, dim * 1, 4, 2)
        x = tf.tanh(dconv(y, 3, 4, 2))
        return x


def D(x, c_dim, dim=64, norm_name='batch_norm', is_training=True):
    norm = _get_norm_fn(norm_name, is_training)
    conv_norm_lrelu = partial(conv, normalizer_fn=norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = conv_norm_lrelu(x, dim, 4, 2)
        y = conv_norm_lrelu(y, dim * 2, 4, 2)
        y = conv_norm_lrelu(y, dim * 4, 4, 2)
        y = conv_norm_lrelu(y, dim * 8, 4, 2)
        logit = fc(y, 1)
        c_logit = fc(y, c_dim)
        return logit, c_logit


# ==============================================================================
# =                                loss function                               =
# ==============================================================================

def get_loss_fn(mode):
    if mode == 'gan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(r_logit), r_logit)
            f_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(f_logit), f_logit)
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(f_logit), f_logit)
            return f_loss

    elif mode == 'lsgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = tf.losses.mean_squared_error(tf.ones_like(r_logit), r_logit)
            f_loss = tf.losses.mean_squared_error(tf.zeros_like(f_logit), f_logit)
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = tf.losses.mean_squared_error(tf.ones_like(f_logit), f_logit)
            return f_loss

    elif mode == 'wgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = - tf.reduce_mean(r_logit)
            f_loss = tf.reduce_mean(f_logit)
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            f_loss = - tf.reduce_mean(f_logit)
            return f_loss

    elif mode == 'hinge':
        def d_loss_fn(r_logit, f_logit):
            r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
            f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
            return r_loss, f_loss

        def g_loss_fn(f_logit):
            # f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
            f_loss = tf.reduce_mean(- f_logit)
            return f_loss

    return d_loss_fn, g_loss_fn


# ==============================================================================
# =                                   others                                   =
# ==============================================================================

def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            with tf.name_scope('interpolate'):
                if b is None:   # interpolation in DRAGAN
                    beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                    _, variance = tf.nn.moments(a, list(range(a.shape.ndims)))
                    b = a + 0.5 * tf.sqrt(variance) * beta
                shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
                alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.get_shape().as_list())
                return inter

        with tf.name_scope('gradient_penalty'):
            x = _interpolate(real, fake)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = tf.gradients(pred, x)[0]
            norm = tf.norm(slim.flatten(grad), axis=1)
            gp = tf.reduce_mean((norm - 1.)**2)
            return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=tf.float32)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)

    return gp


def sample_c(ks, c_1=None, continuous_last=False):
    assert c_1 is None or ks[0] == len(c_1), '`ks[0]` is inconsistent with `c_1`!'

    c_tree = [[np.array([1.])]]
    mask_tree = []

    for l, k in enumerate(ks):
        if c_1 is not None and l == 0:
            c_l = [c_1]
            mask_l = [np.ones_like(c_1)]
        else:
            c_l = []
            mask_l = []
            for i in range(len(c_tree[-1])):
                for j in range(len(c_tree[-1][-1])):
                    if c_tree[-1][i][j] == 1.:
                        if continuous_last is True and l == len(ks) - 1:
                            c_l.append(np.random.uniform(-1, 1, size=[k]))
                        else:
                            c_l.append(np.eye(k)[np.random.randint(k)])
                        mask_l.append(np.ones([k]))
                    else:
                        c_l.append(np.zeros([k]))
                        mask_l.append(np.zeros([k]))
        c_tree.append(c_l)
        mask_tree.append(mask_l)

    c_tree[0:1] = []
    c = np.concatenate([k for l in c_tree for k in l])
    mask = np.concatenate([k for l in mask_tree for k in l])

    return c, mask, c_tree, mask_tree


def traversal_trees(ks, continuous_last=False):
    trees = []
    if len(ks) == 1:
        if continuous_last:
            trees.append([[np.random.uniform(-1, 1, size=[ks[0]])]])
        else:
            for i in range(ks[0]):
                trees.append([[np.eye(ks[0])[i]]])
    else:
        def _merge_trees(trees):
            tree = []
            for l in range(len(trees[0])):
                tree_l = []
                for t in trees:
                    tree_l += t[l]
                tree.append(tree_l)
            return tree

        def _zero_tree(tree):
            zero_tree = []
            for l in tree:
                zero_tree_l = []
                for i in l:
                    zero_tree_l.append(i * 0.)
                zero_tree.append(zero_tree_l)
            return zero_tree

        for i in range(ks[0]):
            trees_i = []
            sub_trees, _ = traversal_trees(ks[1:], continuous_last=continuous_last)
            for j, s_t in enumerate(sub_trees):
                to_merge = [_zero_tree(s_t)] * ks[0]
                to_merge[i] = s_t
                sub_trees[j] = _merge_trees(to_merge)
            for s_t in sub_trees:
                trees_i.append([[np.eye(ks[0])[i]]] + s_t)
            trees += trees_i

    cs = []
    for t in trees:
        cs.append(np.concatenate([k for l in t for k in l]))
    return trees, cs


def to_tree(x, ks):
    size_splits = []
    n_l = 1
    for k in ks:
        for _ in range(n_l):
            size_splits.append(k)
        n_l *= k

    splits = tf.split(x, size_splits, axis=1)

    tree = []
    n_l = 1
    i = 0
    for k in ks:
        tree_l = []
        for _ in range(n_l):
            tree_l.append(splits[i])
            i += 1
        n_l *= k
        tree.append(tree_l)

    return tree


def tree_loss(logits, c, mask, ks, continuous_last=False):
    logits_tree = to_tree(logits, ks)
    c_tree = to_tree(c, ks)
    mask_tree = to_tree(mask, ks)

    losses = []
    for l, logits_l, c_l, mask_l in zip(range(len(logits_tree)), logits_tree, c_tree, mask_tree):
        loss_l = 0
        for lo, c, m in zip(logits_l, c_l, mask_l):
            weights = tf.reduce_mean(m, axis=1)
            if continuous_last is True and l == len(ks) - 1:
                loss_l += tf.losses.mean_squared_error(c, lo, weights=weights)
            else:
                loss_l += tf.losses.softmax_cross_entropy(c, lo, weights=weights)
        losses.append(loss_l)
    return losses

if __name__ == '__main__':
    from pprint import pprint as pp
    s = tf.Session()
    ks = [2, 2, 2]
    c_1 = np.array([1., 0])
    continuous_last = False
    if len(ks) == 1 and c_1 is not None:
        continuous_last = False

    c, mask, c_tree, mask_tree = sample_c(ks, c_1, continuous_last)
    pp(c)
    pp(mask)
    pp(c_tree)
    pp(mask_tree)

    tree = to_tree(tf.constant(np.array([c]), dtype=tf.float32), ks)
    pp(tree)
    pp(s.run(tree))

    pp(tree_loss(tf.constant(np.array([c]), dtype=tf.float32),
                 tf.constant(np.array([c]), dtype=tf.float32),
                 tf.constant(np.array([mask]), dtype=tf.float32),
                 ks,
                 continuous_last))

    pp(s.run(tf.losses.softmax_cross_entropy([[0, 10]], [[0.5, 0.5]])))
    pp(s.run(tf.losses.mean_squared_error([[2, 1]], [[0, 0]])))

    for tree, c in zip(*traversal_trees(ks, continuous_last=continuous_last)):
        pp(tree)
        pp(c)
