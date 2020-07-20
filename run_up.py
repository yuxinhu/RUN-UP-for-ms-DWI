

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from tf_cnnvis import *

import util

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "visualization"])
parser.add_argument("--net", required=True, choices=["unet", "unet3d"])
parser.add_argument("--unroll_opt", required=True, choices=["pocs", "fista"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=20, help="number of images in batch")
parser.add_argument("--num_filters", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--num_conv_layers", type=int, default=3, help="number of convolution layers in each encoder")
parser.add_argument("--num_steps", type=int, default=3, help="number of encoders")
parser.add_argument("--num_unroll_iter", type=int, default=3, help="number of iterations in unrolled optimization")
parser.add_argument("--num_gradient_update", type=int, default=1, help="number of gradient update per iteration")


parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

parser.add_argument("--hardproj", dest="do_hardproj", action="store_true", help="do hard projection in the end of FISTA")
parser.add_argument("--no_do_hardproj", dest="do_hardproj", action="store_false", help="don't do hard projection in the end of FISTA")
parser.set_defaults(do_hardproj=True)

parser.add_argument("--share_weights", dest="share_weights", action="store_true", help="share nn weights across iterations")
parser.add_argument("--not_share_weights", dest="share_weights", action="store_false", help="don't share nn weights across iterations")
parser.set_defaults(share_weights=False)

parser.add_argument("--intermediate_loss", dest="use_intermediate_loss", action="store_true", help="include intermediate loss")
parser.add_argument("--no_use_intermediate_loss", dest="use_intermediate_loss", action="store_false", help="don't include intermediate loss")
parser.set_defaults(use_intermediate_loss=True)

parser.add_argument("--do_batch_norm", dest="batch_norm", action="store_true", help="apply batch norm after each conv layer")
parser.add_argument("--no_batch_norm", dest="batch_norm", action="store_false", help="don't apply batch norm")
parser.set_defaults(batch_norm=False)

parser.add_argument("--do_residual_conv", dest="residual_conv", action="store_true", help="use residual blocks inside each network")
parser.add_argument("--no_residual_conv", dest="residual_conv", action="store_false", help="don't use residual blocks inside each network")
parser.set_defaults(residual_conv=True)

parser.add_argument("--size_x", type=int, default=256, help="number of slices")
parser.add_argument("--size_y", type=int, default=256, help="number of slices")
parser.add_argument("--size_c", type=int, default=8, help="number of slices")

parser.add_argument("--addb0", type=int, default=1, help="whether include b0") # Include b=0 image as the input to the U-Net
parser.add_argument("--oddk", type=int, default=1, help="whether use knet in odd iteration")
parser.add_argument("--evenk", type=int, default=0, help="whether use knet in even iteration") # Otherwise use network in the image space
parser.add_argument("--l2loss", type=int, default=0, help="whether use l2 loss") # Otherwise use L1-loss
parser.add_argument("--lrelu", type=int, default=0, help="whether leaky relu") # Otherwise use PReLU

a = parser.parse_args()

EPS = 1e-12
SIZE_SHOT = 4 # Number of shots
SIZE_C = 8 # Number of coils

Examples = collections.namedtuple("Examples", "paths, ksp, maps, targets, masks, b0s, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, gen_loss_total, gen_loss_final, gen_grads_and_vars, end_points, train")



def lrelu(x, a=0.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)

        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)



def prelu(_x, a=0.2):
  # Parametric ReLU, where a is a variable to be learned.
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def decode_cfl(contents):
    # Extract all data from the cfl file (a long vector).
    k_shape = [a.size_x, a.size_y, SIZE_SHOT, a.size_c] 
    maps_shape = [a.size_x, a.size_y, a.size_c]  ## Yuxin
    im_shape = [a.size_x, a.size_y, SIZE_SHOT]
    mask_shape = [a.size_x, a.size_y, SIZE_SHOT]
    b0_shape = [a.size_x, a.size_y, 1]

    N_kspace = np.prod(k_shape)
    N_maps = np.prod(maps_shape)
    N_im = np.prod(im_shape)
    N_mask = np.prod(mask_shape)
    N_b0 = np.prod(b0_shape)

    N_total = N_kspace + N_maps + N_im + N_mask + N_b0


    with tf.name_scope("decode_data"):
        # Read data from cfl file
        record_bytes = tf.decode_raw(contents, tf.float32)
        # Data is first stored as (..., real/imag)
        image = tf.reshape(record_bytes, [N_total, 2])
        image = tf.complex(image[:, 0], image[:, 1])
        # image = util.channels_to_complex(image)
        image = tf.squeeze(image)

        k_input = tf.reshape(tf.slice(image,[0],[N_kspace]),k_shape[::-1])
        k_input = tf.transpose(k_input, [3, 2, 1, 0])
        maps_input = tf.reshape(tf.slice(image,[N_kspace],[N_maps]),maps_shape[::-1])
        maps_input = tf.transpose(maps_input, [2, 1, 0])
        im_output = tf.reshape(tf.slice(image,[N_kspace + N_maps],[N_im]),im_shape[::-1])
        #im_output = tf.real(im_output)  ## Yuxin
        im_output = tf.transpose(im_output, [2, 1, 0])

        mask = tf.reshape(tf.slice(image,[N_kspace + N_maps + N_im],[N_mask]),mask_shape[::-1])
        mask = tf.transpose(mask, [2, 1, 0])

        b0 = tf.reshape(tf.slice(image, [N_kspace + N_maps + N_im + N_mask], [N_b0]), b0_shape[::-1])
        b0 = tf.transpose(b0, [2, 1, 0])
        return k_input, maps_input, im_output, mask, b0


def load_examples():
    # Data loader
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    
    input_paths = glob.glob(os.path.join(a.input_dir, "*.cfl"))
    decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        k_input, maps_input, target_images, masks, b0 = decode_cfl(contents) #20200204 with b0
    
    # synchronize seed for image operations so that we do the same operations to
    # k-space, sensitivity maps and images    
    seed = random.randint(0, 2**31 - 1)
    
    paths_batch, k_batch, maps_batch, targets_batch, masks_batch, b0_batch = tf.train.batch(
        [paths, k_input, maps_input, target_images, masks, b0], batch_size=a.batch_size) # 20200204 with b0
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        ksp=k_batch,
        maps=maps_batch,
        targets=targets_batch,
        masks=masks_batch,
        b0s = b0_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def create_conv_block(x, out_channel, kernel_size=[3,3], is_training=True, name="conv_block"):
    with tf.variable_scope(name):
        x = slim.conv2d(x, out_channel, kernel_size)
        if a.batch_norm:
            x = slim.batch_norm(x, is_training=is_training)
        x = lrelu(x) if a.lrelu==1 else prelu(x)
    return x

def create_conv_res_block(x, out_channel, kernel_size=[3,3], is_training=True, name="conv_res_block"):
    with tf.variable_scope(name):
        x_orig = x
        x = slim.conv2d(x, out_channel, kernel_size)
        if a.batch_norm:
            x = slim.batch_norm(x, is_training=is_training)
        x = lrelu(x) if a.lrelu==1 else prelu(x)
        x = slim.conv2d(x, out_channel, kernel_size)
        if a.batch_norm:
            x = slim.batch_norm(x, is_training=is_training)
    return x + x_orig

def create_unet(inputs, b0=None, output_channels=None, num_features=[32,64,128], num_conv_layers=3,
               kernel_size=[3, 3], do_residual=True, is_training=True, name="unet", reuse=None):
    #TODO(xinweis): look at why use tf.truncated_normal_initializer, instead of xavier_initializer
    with tf.variable_scope(name, reuse=reuse):
        if output_channels is None:
            output_channels = inputs.get_shape()[-1]

        print('output layer shape:')
        print(inputs.get_shape())
        print(b0.get_shape())

        N_stage = len(num_features)
        stage_out = []
        
        if a.addb0 == 1:
            net = tf.concat(axis=3, values=[inputs, b0])
        else:
            net = inputs
        print(net.get_shape())

        residual_conv=a.residual_conv
        for stage in range(N_stage):
            with tf.variable_scope(('encoder%d' % stage)):
                with slim.arg_scope([slim.conv2d], padding='SAME',
                            activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.001)):


                    if residual_conv:
                        orig_encoder = create_conv_block(net, num_features[stage], kernel_size=kernel_size,
                                                  is_training=is_training,
                                                  name='conv_block')
                        net = orig_encoder
                
                    for k_conv_block in range(num_conv_layers):
                        if residual_conv:
                            net = create_conv_res_block(net, num_features[stage], kernel_size=kernel_size,
                                                      is_training=is_training,
                                                      name='conv_block_res_%d' % k_conv_block)
                        else:
                            net = create_conv_block(net, num_features[stage],
                                                  kernel_size=kernel_size, is_training=is_training,
                                                  name='conv_block_%d' % k_conv_block)
                    if residual_conv:
                        net = net + orig_encoder
                    stage_out.append(net)
                    if stage < N_stage-1:
                        #net = slim.max_pool2d(net, [2, 2], stride=2, padding = 'SAME')
                        net = slim.conv2d(net, num_features[stage], kernel_size = kernel_size, stride = 2, padding = 'SAME')
                        ## toDo: changing to conv with stride (Yuxin)

        # up-sample and decrease depth
        for stage in range(N_stage-1)[::-1]: # one convT and two conv layers in each unit
            print("stage")
            print(stage)
            with tf.variable_scope(('decoder%d' % stage)):
                with slim.arg_scope([slim.conv2d], padding='SAME',
                            activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.001)):
                    net = slim.conv2d_transpose(net, num_features[stage], [4,4], stride=2, 
                                padding='SAME',
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.001))

                    net = tf.concat([net, stage_out[stage]], 3)
                    if a.batch_norm:
                        net = slim.batch_norm(net, is_training=is_training)
                    #net = lrelu(net)
                    net = lrelu(net) if a.lrelu==1 else prelu(net)

                    if residual_conv:
                        orig_decoder = create_conv_block(net, num_features[stage], kernel_size=kernel_size,
                                                  is_training=is_training,
                                                  name='conv_block')
                        net = orig_decoder
                    for k_conv_block in range(num_conv_layers-1):
                        if residual_conv:
                            net = create_conv_res_block(net, num_features[stage],
                                                      kernel_size=kernel_size, is_training=is_training,
                                                      name='conv_block_res_%d' % k_conv_block)
                        else:   
                            net = create_conv_block(net, num_features[stage],
                                                    kernel_size=kernel_size, is_training=is_training,
                                                    name='conv_block_%d' % k_conv_block)
                    if residual_conv:
                        net = orig_decoder + net

        # last layer of conv:
        with tf.variable_scope('output'):
            net = slim.conv2d(net, output_channels, kernel_size,
                              padding='SAME',
                              activation_fn=None,
                              normalizer_fn=None)
            net = slim.conv2d(net, output_channels, kernel_size,
                              padding='SAME',
                              activation_fn=None,
                              normalizer_fn=None)
            if do_residual:
                out = net + inputs
            else:
                out = net

        return out

def unroll_fista(PFx, num_grad_steps=5, num_features=[16,32,32], num_conv_layers=3,
                is_training=True, scope="unrolled",
                name="UnrollFista", 
                sensemap=None, P=None, b0=None, verbose=False):
    """Create FISTA-based unrolled network for MRI.
        PFx: k-space, [batch, x, y, shot, coil]
        P: sampling pattern
    """
    b0k = util.fft2c(b0)
    b0 = util.complex_to_channels(b0)
    b0k = util.complex_to_channels(b0k)

    end_points = {}
    if verbose:
        print("%s> Building FISTA-based unrolled network (%d steps)..."
              % (name, num_grad_steps))
        if sensemap is not None:
            print("%s>  Using sensitivity maps..." % name)

    with tf.variable_scope(scope):
        if P is None:
            P = util.kspace_mask(PFx, dtype=tf.complex64)
        else:
            P = tf.tile(tf.expand_dims(P,-1), [1, 1, 1, 1, a.size_c])

        sensemap = tf.tile(tf.expand_dims(sensemap,-1),[1, 1, 1, 1, SIZE_SHOT])
        sensemap = tf.transpose(sensemap,[0, 1, 2, 4, 3]) ## yuxin
        
        # Init values
        ksx_0 = P * PFx
        print("ksx_0_shape")
        print(ksx_0.get_shape())
        x0 = util.model_transpose(ksx_0, sensemap)

        # Values to be incremented
        ksx_k = ksx_0
        xk = x0
        # Data shapes (complex as channels)
        # shape_xk = xk.shape[-1].value * 2

        for i_step in range(num_grad_steps):
            iter_name = "iter_%02d" % i_step
            print(iter_name)
            # update
            with tf.variable_scope(iter_name):
                for grad_step in range(a.num_gradient_update): # sarah: multiple gradient update before unet
                    if grad_step != 0:
                        xk = util.channels_to_complex(xk)
                    xk_orig = xk
                    # xk = tf.complex(xk, tf.zeros_like(xk))
                    xk = tf.cast(xk, tf.complex64)
                    ksx_k = util.model_forward(xk, sensemap)
                    ksx_k = P * ksx_k
                    xk = util.model_transpose(ksx_k, sensemap)
                    xk = xk - x0
                    # t = tf.get_variable("t", dtype=tf.complex64,
                    #                    initializer=tf.constant([-2.0 + 0.j],dtype = tf.complex64))
                    if grad_step == 0:
                        t = tf.get_variable("t", dtype=tf.float32,
                                        initializer=tf.constant([-2.0]))
                    xk = util.complex_to_channels(xk)
                    xk_orig = util.complex_to_channels(xk_orig)
                    xk = xk_orig + t * xk
                end_points[iter_name + "_grad"] = xk

            # if (i_step != 0):
            #     xk = util.complex_to_channels(xk)
            # for k-net
            if (i_step % 2 == 0 and a.oddk == 1) or (i_step % 2 == 1 and a.evenk == 1):
                xk = util.channels_to_complex(xk)
                xk = util.fft2c(xk)
                xk = util.complex_to_channels(xk)        

            # prox 

            if a.oddk == a.evenk:
                reuse = None if i_step==0 else a.share_weights
                name = a.net if a.share_weights else (a.net + "-" + iter_name)
            else:
                reuse = None if i_step<=1 else a.share_weights
                if a.share_weights:
                    name = a.net + "iter_%02d" % (i_step%2)
                else:
                    name = a.net + "-" + iter_name

            print("input_shape_unet")
            print(xk.get_shape())

            if (i_step % 2 == 0 and a.oddk == 1) or (i_step % 2 == 1 and a.evenk == 1):
                xk = create_unet(
                    xk, b0 = b0k, output_channels=SIZE_SHOT * 2,
                    num_features=num_features, num_conv_layers=num_conv_layers,
                    is_training=is_training, name=name, reuse=reuse)
            else:
                xk = create_unet(
                    xk, b0 = b0, output_channels=SIZE_SHOT * 2,
                    num_features=num_features, num_conv_layers=num_conv_layers,
                    is_training=is_training, name=name, reuse=reuse)

            xk = util.channels_to_complex(xk)

            if (i_step % 2 == 0 and a.oddk == 1) or (i_step % 2 == 1 and a.evenk == 1):
                print("ifft2c")
                xk = util.ifft2c(xk)
            end_points[iter_name] = xk
        
        if a.do_hardproj:
            if verbose:
                print("%s>   Final hard data projection..." % name)
            # Final data projection
            xk = tf.complex(xk[:, 0], xk[:, 1]) # yuxin: transfer output from real to complex
            ksx_k = util.model_forward(xk, sensemap)
            ksx_k = P * ksx_0 + (1 - P) * ksx_k
            xk = util.model_transpose(ksx_k, sensemap)

    end_points['inputs'] = x0
    end_points['output-ks'] = ksx_k
    return xk, end_points



def create_model(ksp, maps, targets, masks, b0, is_training):
    num_features = []
    
    for k in range(a.num_steps):
        num_features.append(a.num_filters*(2**k if k<=1 else 1)) # sarah: changing to alway increase depth 
    # num_features.append(num_features[-1])

    outputs, end_points = unroll_fista(ksp, num_grad_steps=a.num_unroll_iter,
                num_features=num_features, num_conv_layers=a.num_conv_layers, is_training=is_training,
                sensemap=maps, verbose=True, P=masks, b0=b0)

    print("outputs shape")
    print(outputs.get_shape())

    
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        if a.l2loss == 1:
            gen_loss_final = tf.nn.l2_loss(tf.abs(targets - outputs))
        else:
            gen_loss_final = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_total = gen_loss_final
        if a.use_intermediate_loss:
            for i_step in range(a.num_steps):
                iter_name = "iter_%02d" % i_step
                if a.l2loss == 1:
                    gen_loss_total += tf.nn.l2_loss(tf.abs(targets - end_points[iter_name]))
                else:
                    gen_loss_total += tf.reduce_mean(tf.abs(targets - end_points[iter_name]))


    with tf.name_scope("generator_train"): 
        gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unrolled')

        gen_optim = tf.train.AdamOptimizer(learning_rate=a.lr, beta1=a.beta1, epsilon=0.00001)
        gen_train = slim.learning.create_train_op(gen_loss_total, gen_optim)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss_total, var_list=gen_tvars)
        # gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    if a.use_intermediate_loss:
        update_losses = ema.apply([gen_loss_total, gen_loss_final])
    else:
        update_losses = ema.apply([gen_loss_total])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        gen_loss_total=ema.average(gen_loss_total),
        gen_loss_final=ema.average(gen_loss_final),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        end_points=end_points,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
        h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        # for kind in ["inputs", "outputs", "targets"]:
        for kind in ["outputs", "targets"]:
            filename = name + "-" + kind
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            out_path = os.path.join(image_dir, filename)
            # print(out_path)
            contents = fetches[kind][i]
            
            # contents = contents[...,0] + 1j*contents[...,1]
            # print(contents.dtype)
            # print(kind)
            # print(np.amax(contents[:,12,:], axis=1))
            writecfl(out_path, contents)
        # for kind in ["iters"]:
        #     for key, value in fetches[kind].iteritems():
        #         filename = name + "-" + key
        #         out_path = os.path.join(image_dir, filename)
        #         writecfl(out_path, value)
           


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "visualization":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "net", "num_filters", "num_steps"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)


    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.ksp, examples.maps, examples.targets, examples.masks, examples.b0s, a.mode=="train")

    inputs = model.end_points["inputs"]
    targets = examples.targets
    outputs = model.outputs
    im_iters = model.end_points

    # print("inputs tensor datatype" + str(inputs.dtype))
    # print("targets tensor datatype" + str(targets.dtype))
    # print("outputs tensor datatype" + str(outputs.dtype))

    display_fetches = {
        "paths": examples.paths,
        "inputs": inputs,
        "targets": targets,
        "outputs": outputs,
        "iters": im_iters,
    }

    # summaries
    # input, output, target and difference
    inputs_sos = tf.transpose(util.sumofsq(inputs, keep_dims=True, axis=3), [0, 2, 1, 3])
    outputs_sos = tf.transpose(util.sumofsq(outputs, keep_dims=True, axis=3), [0, 2, 1, 3])
    targets_sos = tf.transpose(util.sumofsq(targets, keep_dims=True, axis=3), [0, 2, 1, 3])

    image_cat = tf.concat([inputs_sos, outputs_sos, targets_sos], 1)
    image_cat2 = tf.abs(tf.concat([inputs_sos-targets_sos, outputs_sos-targets_sos, targets_sos-targets_sos], 1))

    with tf.name_scope("results_summary"):
        tf.summary.image("input/output/target", tf.concat([image_cat, image_cat2], 2))

    # results during iterations
    for i_step in range(a.num_unroll_iter):
        iter_name = "iter_%02d" % i_step
        tmp = tf.transpose(util.sumofsq(im_iters[iter_name], keep_dims=True, axis=3), [0, 2, 1, 3])
        if i_step == 0:
            iterations_sos = tmp
        else:
            iterations_sos = tf.concat([iterations_sos, tmp], 1)

    with tf.name_scope("iterations_summary"):
        tf.summary.image("iterations", iterations_sos)

    tf.summary.scalar("generator_loss_total", model.gen_loss_total)
    tf.summary.scalar("generator_loss_final", model.gen_loss_final)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=5)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

    if a.mode == "test" or a.mode == "train":
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            print("parameter_count =", sess.run(parameter_count))

            if a.checkpoint is not None:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(a.checkpoint)
                saver.restore(sess, checkpoint)

            max_steps = 2**32
            if a.max_epochs is not None:
                max_steps = examples.steps_per_epoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps

            if a.mode == "test":
                # testing
                # at most, process the test data once
                max_steps = min(examples.steps_per_epoch, max_steps)
                total_time = 0
                for step in range(max_steps):
                    start = time.time()
                    results = sess.run(display_fetches)
                    total_time += (time.time() - start)
                    save_images(results)
                print("total time = ", total_time)
            elif a.mode == "train":
                # training
                start = time.time()

                for step in range(max_steps):
                    def should(freq):
                        return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                    options = None
                    run_metadata = None
                    if should(a.trace_freq):
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                    fetches = {
                        "train": model.train,
                        "global_step": sv.global_step,
                    }

                    if should(a.progress_freq):
                        fetches["gen_loss_final"] = model.gen_loss_final
                        fetches["gen_loss_total"] = model.gen_loss_total

                    if should(a.summary_freq):
                        fetches["summary"] = sv.summary_op

                    if should(a.display_freq):
                        fetches["display"] = display_fetches

                    results = sess.run(fetches, options=options, run_metadata=run_metadata)

                    if should(a.summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], results["global_step"])

                    if should(a.display_freq):
                        print("saving display images")
                        filesets = save_images(results["display"], step=results["global_step"])
                        append_index(filesets, step=True)

                    if should(a.trace_freq):
                        print("recording trace")
                        sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                    if should(a.progress_freq):
                        # global_step will have the correct step count if we resume from a checkpoint
                        train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                        train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                        rate = (step + 1) * a.batch_size / (time.time() - start)
                        remaining = (max_steps - step) * a.batch_size / rate
                        print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                        print("gen_loss_final", results["gen_loss_final"])
                        print("gen_loss_total", results["gen_loss_total"])

                    if should(a.save_freq):
                        print("saving model")
                        saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                    if sv.should_stop():
                        break


main()

