#!/usr/bin/python3
# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
from __future__ import division
from __future__ import print_function

from IPython import embed

from config import cfg
import network_desp
import dataset

import tensorflow.contrib.slim as slim
import tensorflow as tf
import sys, os, pprint, time, glob, argparse, logging, setproctitle, numpy as np

from utils.dpflow.data_provider import DataFromList, MultiProcessMapDataZMQ
from utils.dpflow.prefetching_iter import PrefetchingIter
from utils.tf_utils.model_helper import average_gradients, sum_gradients, \
    get_variables_in_checkpoint_file

from tqdm import tqdm
from utils.py_utils import QuickLogger, misc
from utils.py_faster_rcnn_utils.timer import Timer


def snapshot(sess, saver, epoch):
    filename = 'epoch_{:d}'.format(epoch) + '.ckpt'
    model_dump_dir = os.path.join(cfg.output_dir, 'model_dump')
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)
    filename = os.path.join(model_dump_dir, filename)
    saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))


def get_data_flow():
    source = cfg.train_source
    with open(source) as f:
        files = f.readlines()

    data = DataFromList(files)
    dp = MultiProcessMapDataZMQ(
        data, cfg.nr_dataflow, dataset.get_data_for_singlegpu)
    dp.reset_state()
    dataiter = dp.get_data()
    return dataiter


def train(args):
    logger = QuickLogger(log_dir=cfg.output_dir).get_logger()
    logger.info(cfg)
    np.random.seed(cfg.rng_seed)
    # GPU setting
    num_gpu = len(args.devices.split(','))
    # Network defintion
    net = network_desp.Network()
    # Loading training data
    data_iter = get_data_flow()
    prefetch_data_layer = PrefetchingIter(data_iter, num_gpu)

    with tf.Graph().as_default(), tf.device('/device:CPU:0'):
        # ??
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0.),
            trainable=False)

        tfconfig = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        tf.set_random_seed(cfg.rng_seed)

        # Learning rate
        lr = tf.Variable(cfg.get_lr(0), trainable=False) # ?
        lr_placeholder = tf.placeholder(lr.dtype, shape=lr.get_shape()) # ?
        update_lr_op = lr.assign(lr_placeholder)
        # What is momentum?
        opt = tf.train.MomentumOptimizer(lr, cfg.momentum)

        '''data processing'''
        # List of tf placceholder for input: 
        # data(image), im_info(2x6: [resize_h, resize_h, ??, h, w, ??]), bbox
        inputs_list = []
        for i in range(num_gpu):
            # Get list of input placeholder
            inputs_list.append(net.get_inputs())
        put_op_list = []
        get_op_list = []
        for i in range(num_gpu):
            with tf.device("/GPU:%s" % i):
                # ????
                area = tf.contrib.staging.StagingArea(
                    dtypes=[tf.float32 for _ in range(len(inputs_list[0]))])
                put_op_list.append(area.put(inputs_list[i]))
                get_op_list.append(area.get())
        # Coordinator to handle multiple threads
        coord = tf.train.Coordinator()
        # Initialize all variables
        init_all_var = tf.initialize_all_variables()
        sess.run(init_all_var)
        # Starts all queue runners collected in the graph
        queue_runner = tf.train.start_queue_runners(coord=coord, sess=sess)
        '''end of data processing'''

        tower_grads = []
        # Parameter regulazier
        biases_regularizer = tf.no_regularizer
        biases_ini = tf.constant_initializer(0.0)
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.weight_decay)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i):
                        with slim.arg_scope(
                                [slim.model_variable, slim.variable],
                                device='/device:CPU:0'):
                            with slim.arg_scope(
                                    [slim.conv2d, slim.conv2d_in_plane,
                                     slim.conv2d_transpose,
                                     slim.separable_conv2d,
                                     slim.fully_connected],
                                    weights_regularizer=weights_regularizer,
                                    biases_regularizer=biases_regularizer,
                                    biases_initializer=biases_ini):
                                # Learning loss
                                loss = net.inference('TRAIN', get_op_list[i])
                                loss = loss / num_gpu
                                if i == num_gpu - 1:
                                    # Regularization loss
                                    regularization_losses = tf.get_collection(
                                        tf.GraphKeys.REGULARIZATION_LOSSES)
                                    loss = loss + tf.add_n(
                                        regularization_losses)

                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        if len(tower_grads) > 1:
            grads = sum_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        final_gvs = []
        # Why do gradient multiplication of 3
        with tf.variable_scope('Gradient_Mult'):
            for grad, var in grads:
                scale = 1.
                # if '/biases:' in var.name:
                #    scale *= 2.
                if 'conv_new' in var.name:
                    scale *= 3.
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gvs.append((grad, var))
        apply_gradient_op = opt.apply_gradients(
            final_gvs, global_step=global_step)

        # What is moving average
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # apply_gradient_op = opt.apply_gradients(grads)

        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=100000)
        saver = tf.train.Saver(max_to_keep=100000)

        variables = tf.global_variables()
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(update_lr_op, {lr_placeholder: cfg.get_lr(0) * num_gpu})
        sess.run(tf.variables_initializer(variables, name='init'))

        # Restore existing model
        var_keep_dic = get_variables_in_checkpoint_file(cfg.weight)
        var_keep_dic.pop('global_step')
        variables_to_restore = []
        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                # print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, cfg.weight)

        # Train collection(tensor): rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, tot_losses
        train_collection = net.get_train_collection()
        sess2run = []
        sess2run.append(train_op)
        sess2run.append(put_op_list)
        for col in train_collection.values():
            sess2run.append(col)

        timer = Timer()

        # warm up staging area
        # Get input list only with names
        inputs_names = net.get_inputs(mode=1)
        for _ in range(4):
            blobs_list = prefetch_data_layer.forward()
            feed_dict = {}
            for i, inputs in enumerate(inputs_list):
                # blobs = next(data_iter)
                blobs = blobs_list[i]
                for it_idx, it_inputs_name in enumerate(inputs_names):
                    feed_dict[inputs[it_idx]] = blobs[it_inputs_name]
            sess.run([put_op_list], feed_dict=feed_dict)

        # Warmup: max_epoch 30
        for epoch in range(cfg.max_epoch):
            # Only with first epoch, warm_iter = 500
            #if epoch == 0 and cfg.warm_iter > 0:
            if 0 and epoch == 0 and cfg.warm_iter > 0:
                pbar = tqdm(range(cfg.warm_iter))
                # up_lr=0.00125, bottom_lr=0.000416, iter_delta_lr=0.00000166
                up_lr = cfg.get_lr(0) * num_gpu
                bottom_lr = up_lr * cfg.warm_fractor
                iter_delta_lr = 1.0 * (up_lr - bottom_lr) / cfg.warm_iter
                cur_lr = bottom_lr
                for iter in pbar:
                    # Update learning rate, must through session run?
                    sess.run(update_lr_op, {lr_placeholder: cur_lr})
                    cur_lr += iter_delta_lr
                    feed_dict = {}
                    # Get batch data from prefetch_data_layer: one batch contains Two images
                    # blobs_list={"is_valid', 'data'(2x750x1333x3), 'im_info', 'boxes'}
                    # "im_info": image size info for two images in the batch: 
                    #   [[train_image_w, train_image_h, scale, orig_image_w, orig_image_h, ?], []]
                    #   [[750, 1333, 0.694, 1080, 1920, 4], [750, 1333, 0.694, 1080, 1920, 7]]
                    # "data": image data, shape=[2x750x1333x3]
                    # "boxes": gt bbox of two images, shape=[2x100x5], 1x5=[xmin,ymin,xmax,ymax,type_id]
                    #          the top N items from odgt, the rest is 0
                    # "is_valid": whether to use it
                    blobs_list = prefetch_data_layer.forward()
                    # inputs_list: 3 placeholder: image_data, im_info, bbox
                    # inputs_names: ['data', 'im_info', 'boxes']
                    for i, inputs in enumerate(inputs_list):
                        # blobs = next(data_iter)
                        blobs = blobs_list[i]
                        for it_idx, it_inputs_name in enumerate(inputs_names):
                            feed_dict[inputs[it_idx]] = blobs[it_inputs_name]
                    # Get all losses
                    sess_ret = sess.run(sess2run, feed_dict=feed_dict)

                    print_str = 'iter %d, ' % (iter)
                    for idx_key, iter_key in enumerate(train_collection.keys()):
                        print_str += iter_key + ': %.4f, ' % sess_ret[
                            idx_key + 2]

                    print_str += 'lr: %.4f, speed: %.3fs/iter' % \
                                 (cur_lr, timer.average_time)
                    logger.info(print_str)
                    pbar.set_description(print_str)

            # Real training stage
            # nr_image_per_epoch: 80000, train_batch_per_gpu: 2
            # "//": integer division
            pbar = tqdm(range(1, cfg.nr_image_per_epoch // \
                              (num_gpu * cfg.train_batch_per_gpu) + 1))
            cur_lr = cfg.get_lr(epoch) * num_gpu
            sess.run(update_lr_op, {lr_placeholder: cur_lr})
            logger.info("epoch: %d" % epoch)
            for iter in pbar:
                timer.tic()
                feed_dict = {}
                blobs_list = prefetch_data_layer.forward()
                for i, inputs in enumerate(inputs_list):
                    # blobs = next(data_iter)
                    blobs = blobs_list[i]
                    for it_idx, it_inputs_name in enumerate(inputs_names):
                        feed_dict[inputs[it_idx]] = blobs[it_inputs_name]

                sess_ret = sess.run(sess2run, feed_dict=feed_dict)
                timer.toc()

                print_str = 'iter %d, ' % (iter)
                for idx_key, iter_key in enumerate(train_collection.keys()):
                    print_str += iter_key + ': %.4f, ' % sess_ret[idx_key + 2]

                print_str += 'lr: %.4f, speed: %.3fs/iter' % \
                             (cur_lr, timer.average_time)
                logger.info(print_str)
                pbar.set_description(print_str)

            if epoch%3 == 0:
              snapshot(sess, saver, epoch)
        coord.request_stop()
        coord.join(queue_runner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test network')
    parser.add_argument(
        '-d', '--devices', default='0', type=str, help='device for training')
    args = parser.parse_args()
    args.devices = misc.parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    setproctitle.setproctitle('train ' + cfg.this_model_dir)
    train(args)
