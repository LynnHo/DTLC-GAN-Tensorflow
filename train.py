from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback

import data
import imlib as im
import model
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--att', dest='att', default='', choices=list(data.Celeba.att_dict.keys()) + [''])
parser.add_argument('--ks', dest='ks', type=int, default=[2, 3, 3], nargs='+', help='k each layer')
parser.add_argument('--lambdas', dest='lambdas', type=float, default=[1., 1., 1.], nargs='+', help='loss weight of each layer')
parser.add_argument('--continuous_last', dest='continuous_last', action='store_true')
parser.add_argument('--half_acgan', dest='half_acgan', action='store_true')

parser.add_argument('--epoch', dest='epoch', type=int, default=100)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.0002, help='learning rate of d')
parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.0002, help='learning rate of g')
parser.add_argument('--n_d', dest='n_d', type=int, default=1)
parser.add_argument('--n_g', dest='n_g', type=int, default=1)
parser.add_argument('--n_d_pre', dest='n_d_pre', type=int, default=0)
parser.add_argument('--optimizer', dest='optimizer', default='adam', choices=['adam', 'rmsprop'])

parser.add_argument('--z_dim', dest='z_dim', type=int, default=100, help='dimension of latent')
parser.add_argument('--loss_mode', dest='loss_mode', default='gan', choices=['gan', 'lsgan', 'wgan', 'hinge'])
parser.add_argument('--gp_mode', dest='gp_mode', default='none', choices=['none', 'dragan', 'wgan-gp'], help='type of gradient penalty')
parser.add_argument('--norm', dest='norm', default='batch_norm', choices=['batch_norm', 'instance_norm', 'layer_norm', 'none'])

parser.add_argument('--experiment_name', dest='experiment_name', default='default')

args = parser.parse_args()

att = args.att
ks = args.ks
if att != '':
    ks[0] = 2
lambdas = args.lambdas
assert len(ks) == len(lambdas), 'The lens of `ks` and `lambdas` should be the same!'
continuous_last = args.continuous_last
if len(ks) == 1 and att != '':
    continuous_last = False
half_acgan = args.half_acgan

epoch = args.epoch
batch_size = args.batch_size
lr_d = args.lr_d
lr_g = args.lr_g
n_d = args.n_d
n_g = args.n_g
n_d_pre = args.n_d_pre
optimizer = args.optimizer

z_dim = args.z_dim
loss_mode = args.loss_mode
gp_mode = args.gp_mode
norm = args.norm

experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

img_size = 64

# dataset
dataset = data.Celeba('./data', ['Bangs' if att == '' else att], img_size, batch_size)


# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

# models
c_dim = len(model.sample_c(ks)[0])
D = partial(model.D, c_dim=c_dim, norm_name=norm)
G = model.G

# otpimizer
if optimizer == 'adam':
    optim = partial(tf.train.AdamOptimizer, beta1=0.5)
elif optimizer == 'rmsprop':
    optim = tf.train.RMSPropOptimizer

# loss func
d_loss_fn, g_loss_fn = model.get_loss_fn(loss_mode)
tree_loss_fn = partial(model.tree_loss, ks=ks, continuous_last=continuous_last)

# inputs
real = tf.placeholder(tf.float32, [None] + [img_size, img_size, 3])
z = tf.placeholder(tf.float32, [None, z_dim])
c = tf.placeholder(tf.float32, [None, c_dim])
mask = tf.placeholder(tf.float32, [None, c_dim])

counter = tf.placeholder(tf.int64, [])
layer_mask = tf.constant(np.tril(np.ones(len(ks))), dtype=tf.float32)[counter // (epoch // len(ks))]

# generate
fake = G(z, c)

# dicriminate
r_logit, r_c_logit = D(real)
f_logit, f_c_logit = D(fake)

# d loss
d_r_loss, d_f_loss = d_loss_fn(r_logit, f_logit)
d_f_tree_losses = tree_loss_fn(f_c_logit, c, mask)
if att != '':
    d_r_tree_losses = tree_loss_fn(r_c_logit, c, mask)
    start = 1 if half_acgan else 0
    d_tree_loss = sum([d_f_tree_losses[i] * lambdas[i] * layer_mask[i] for i in range(start, len(lambdas))])
    d_tree_loss += d_r_tree_losses[0] * lambdas[0] * layer_mask[0]
else:
    d_tree_loss = sum([d_f_tree_losses[i] * lambdas[i] for i in range(len(lambdas))])
gp = model.gradient_penalty(D, real, fake, gp_mode)
d_loss = d_r_loss + d_f_loss + d_tree_loss + gp * 10.0

# g loss
g_f_loss = g_loss_fn(f_logit)
g_f_tree_losses = tree_loss_fn(f_c_logit, c, mask)
g_tree_loss = sum([g_f_tree_losses[i] * lambdas[i] * layer_mask[i] for i in range(len(lambdas))])
g_loss = g_f_loss + g_tree_loss

# optims
d_step = optim(learning_rate=lr_d).minimize(d_loss, var_list=tl.trainable_variables(includes='D'))
g_step = optim(learning_rate=lr_g).minimize(g_loss, var_list=tl.trainable_variables(includes='G'))

# summaries
d_summary = tl.summary({d_r_loss: 'd_r_loss',
                        d_f_loss: 'd_f_loss',
                        d_r_loss + d_f_loss: 'd_loss',
                        gp: 'gp'}, scope='D')
tmp = {l: 'd_f_tree_loss_%d' % i for i, l in enumerate(d_f_tree_losses)}
if att != '':
    tmp.update({d_r_tree_losses[0]: 'd_r_tree_loss_0'})
d_tree_summary = tl.summary(tmp, scope='D_Tree')
d_summary = tf.summary.merge([d_summary, d_tree_summary])

g_summary = tl.summary({g_f_loss: 'g_f_loss'}, scope='G')
g_tree_summary = tl.summary({l: 'g_f_tree_loss_%d' % i for i, l in enumerate(g_f_tree_losses)}, scope='G_Tree')
g_summary = tf.summary.merge([g_summary, g_tree_summary])

# sample
z_sample = tf.placeholder(tf.float32, [None, z_dim])
c_sample = tf.placeholder(tf.float32, [None, c_dim])
f_sample = G(z_sample, c_sample, is_training=False)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# epoch counter
ep_cnt, update_cnt = tl.counter(start=1)

# session
sess = tl.session()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    c_ipt_sample = np.stack(model.traversal_trees(ks, continuous_last=continuous_last)[1])
    z_ipt_samples = [np.stack([np.random.normal(size=[z_dim])] * len(c_ipt_sample)) for i in range(15)]

    it = 0
    it_per_epoch = len(dataset) // (batch_size * n_d)
    for ep in range(sess.run(ep_cnt), epoch + 1):
        sess.run(update_cnt)

        dataset.reset()
        for i in range(it_per_epoch):
            it += 1

            # train D
            if n_d_pre > 0 and it <= 25:
                n_d_ = n_d_pre
            else:
                n_d_ = n_d
            for _ in range(n_d_):
                # batch data
                real_ipt, att_ipt = dataset.get_next()
                c_ipt = []
                mask_ipt = []
                for idx in range(batch_size):
                    if att == '':
                        c_1 = None
                    else:
                        if att_ipt[idx] == 1:
                            c_1 = np.array([1.0, 0])
                        else:
                            c_1 = np.array([0, 1.0])
                    c_tmp, mask_tmp, _, _ = model.sample_c(ks, c_1, continuous_last)
                    c_ipt.append(c_tmp)
                    mask_ipt.append(mask_tmp)
                c_ipt = np.stack(c_ipt)
                mask_ipt = np.stack(mask_ipt)
                z_ipt = np.random.normal(size=[batch_size, z_dim])

                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt, c: c_ipt, mask: mask_ipt, counter: ep})
                summary_writer.add_summary(d_summary_opt, it)

            # train G
            for _ in range(n_g):
                # batch data
                z_ipt = np.random.normal(size=[batch_size, z_dim])

                g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt, c: c_ipt, mask: mask_ipt, counter: ep})
                summary_writer.add_summary(g_summary_opt, it)

            # display
            if it % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (ep, i + 1, it_per_epoch))

            # sample
            if it % 100 == 0:
                merge = []
                for z_ipt_sample in z_ipt_samples:
                    f_sample_opt = sess.run(f_sample, feed_dict={z_sample: z_ipt_sample, c_sample: c_ipt_sample}).squeeze()

                    k_prod = 1
                    for k in ks:
                        k_prod *= k
                        f_sample_opts_k = list(f_sample_opt)
                        for idx in range(len(f_sample_opts_k)):
                            if idx % (len(f_sample_opts_k) / k_prod) != 0:
                                f_sample_opts_k[idx] = np.zeros_like(f_sample_opts_k[idx])
                        merge.append(np.concatenate(f_sample_opts_k, axis=1))
                merge = np.concatenate(merge, axis=0)

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(merge, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, it_per_epoch))

        save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
        print('Model is saved in file: %s' % save_path)
except:
    traceback.print_exc()
finally:
    sess.close()
