from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import config as cfg
import time
from tensorflow.python import pywrap_tensorflow
from net import inference_multiview_feature_aggregation

from loss_function import weighted_cross_entropy_loss
from image_reader_sequence_near import ImageReader

import cv2


os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_gpu

IMAGE_HEIGHT = cfg.IMAGE_HEIGHT
IMAGE_WIDTH = cfg.IMAGE_WIDTH
power = cfg.decay_rate
train_number = cfg.train_number
global_step = cfg.total_iter
weight_decay = cfg.weight_decay
batch_size = cfg.batch_size

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if v.name.split(':')[0] in var_keep_dic and 'Variable' not in v.name.split(':')[0]:
                # and 'logits' not in v.name.split(':')[0] and 'Variable' not in v.name.split(':')[0]:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
    return variables_to_restore

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def main(argv=None):
    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    # Train
    print('Train ' + cfg.train_data_list)

    print('RueMonge reader')
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            cfg.train_data_dir,
            cfg.train_data_list,
            input_size,
            cfg.random_scale,
            cfg.random_resize,
            cfg.random_mirror,
            cfg.random_color,
            cfg.random_crop_pad,
            cfg.ignore_label,
            cfg.IMG_MEAN,
            coord)
        image_batch, label_block, occ, label_batch = reader.dequeue(cfg.batch_size)

    # Define Network

    pred_label, logits, logits_refined, _, _, refine_loss_map = \
        inference_multiview_feature_aggregation(image_batch, is_training=True)

    k = int(cfg.seq_num / 2)

    label_occ = label_batch - label_block       # For occluded regions
    logits_loss = tf.constant(0., tf.float32)


    # 3 multi-view attention loss for aggregation module
    for key, logit in enumerate(logits):
        for i in range(len(logit)):
            if i != key:
                region_gt = tf.cast(label_occ[:, :, :, key:key + 1] * occ[:, :, :, i:i + 1], dtype=tf.int32)
                logits_loss += weighted_cross_entropy_loss(logit[i], region_gt)
            else:
                logits_loss += weighted_cross_entropy_loss(logit[i], label_block[:, :, :, key:key + 1])

    # multi-view-loss
    for i in range(len(logits_refined)):
        logits_loss += weighted_cross_entropy_loss(logits_refined[i], label_batch[:, :, :, k:k + 1])
    print('Loss: Rue')

    ce_loss = logits_loss # cross entropy loss


    l2_loss = [weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'weights' or 'w' in v.name or 'W' in v.name]      # encode: W, facade: weights
    l2_losses = tf.add_n(l2_loss)

    # Total loss
    loss = ce_loss + l2_losses

    tf.summary.scalar("loss_ce", ce_loss)
    tf.summary.scalar("l2_losses", l2_losses)
    tf.summary.scalar("total_loss", loss)

    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    # Using Poly learning rate policy
    base_lr = tf.constant(cfg.learning_rate)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / global_step), power))


    trainable_var = tf.trainable_variables()

    # Optimizer
    if cfg.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
        print('Optimizer: Adam')
    elif cfg.optimizer == 'Adam2':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99)
    elif cfg.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif cfg.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        print('Optimizer: Momentum')
    elif cfg.optimizer == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    ## Optimizer definition - nothing different from any classical example
    opt = optimizer

    ## Retrieve all trainable variables you defined in your graph
    # tvs = tf.trainable_variables()
    if cfg.freeze_bn:
        tvs = [v for v in tf.trainable_variables()
               if 'beta' not in v.name and 'gamma' not in v.name]
    else:
        tvs = [v for v in tf.trainable_variables()]

    # # Only train fusion or aggregation module
    # tvs = [v for v in tf.trainable_variables()
    #        if 'fusion' in v.name]

    ## Creation of a list of variables with the same shape as the trainable ones
    # initialized with 0s
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

    ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
    gvs = opt.compute_gradients(loss, tvs)

    ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    ## Define the training step (part with variable value update)
    train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # Set gpu usage
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    sess = tf.Session(config=config)
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=cfg.model_save_num)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    if not os.path.exists(cfg.logs_dir):
        os.makedirs(cfg.logs_dir)
    train_writer = tf.summary.FileWriter(cfg.logs_dir + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(cfg.logs_dir + 'test')

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    count = 0
    files = os.path.join(cfg.save_dir + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) > 0:
        sess.run(tf.global_variables_initializer())
        sfile = glob.glob(files)
        steps = []
        for s in sfile:
            part = s.split('.')
            step = int(part[1].split('-')[1])
            steps.append(step)
        count = max(steps)
        model = cfg.save_dir + 'model.ckpt-' + str(count)
        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')
    else:
        # # restore from pre-train on imagenet
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))

        # # tensorflow                          1
        if os.path.exists(cfg.pre_trained_model) or os.path.exists(cfg.pre_trained_model + '.index'):
            var_keep_dic = get_variables_in_checkpoint_file(cfg.pre_trained_model)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            if len(variables_to_restore) > 0:
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, cfg.pre_trained_model)
                print('Model pre-train loaded from ' + cfg.pre_trained_model)
            else:
                print(cfg.pre_trained_model + ' no params to load.')
        else:
            print('Model inited random.')

    key = int(cfg.seq_num / 2)
    _mask = pred_label[0]
    _img = image_batch[0, :, :, int(key * 3): int((key+1)*3)]
    _gt = label_batch[0, :, :, key: key+1]

    if not os.path.exists(cfg.save_dir + 'temp_img'):
        os.mkdir(cfg.save_dir + 'temp_img')

    print('Start train ' + cfg.data_dir)
    print('---------------Hyper Paras---------------')
    print('-- batch_size: ', cfg.batch_size)
    print('-- Gradient Accumulation: ', cfg.Gradient_Accumulation)
    print('-- image height: ', cfg.IMAGE_HEIGHT)
    print('-- image width: ', cfg.IMAGE_WIDTH)
    print('-- learning rate: ', cfg.learning_rate)
    print('-- GPU: ', cfg.use_gpu)
    print('-- optimizer: ', cfg.optimizer)
    print('-- class num: ', cfg.NUM_OF_CLASSESS)
    print('-- total iter: ', cfg.total_iter)
    print('-- Time acc: ' , cfg.is_time_acc)
    print('-- Acc interval: ', cfg.acc_interval)
    print('-- Start Acc iter: ', cfg.start_show_iter)
    print('-- Is save step: ', cfg.is_save_step)
    print('-- Start save step: ', cfg.start_save_step)
    print('-- save ecpoch: ', cfg.save_step_inter)
    print('-- model save num: ', cfg.model_save_num)
    print('-- summary interval: ', cfg.summary_interval)
    print('-- weight decay: ', cfg.weight_decay)
    print('-- Freeze BN: ', cfg.freeze_bn)
    print('-- Decay rate: ', cfg.decay_rate)
    print('-- minScale: ', cfg.minScale)
    print('-- maxScale: ', cfg.maxScale)
    print('-- random scale: ', cfg.random_scale)
    print('-- random mirror: ', cfg.random_mirror)
    print('-- random crop: ', cfg.random_crop_pad)
    print('-- Validation on :' + str(cfg.val_data_list))
    print('----------------End---------------------')
    fcfg = open(cfg.save_dir + 'cfg.txt', 'w')
    fcfg.write('-- batch_size: ' + str(cfg.batch_size) + '\n')
    fcfg.write('-- Gradient Accumulation: ' + str(cfg.Gradient_Accumulation) + '\n')
    fcfg.write('-- image height: ' + str(cfg.IMAGE_HEIGHT) + '\n')
    fcfg.write('-- image width: ' + str(cfg.IMAGE_WIDTH) + '\n')
    fcfg.write('-- learning rate: ' + str(cfg.learning_rate) + '\n')
    fcfg.write('-- GPU: ' + str(cfg.use_gpu) + '\n')
    fcfg.write('-- optimizer: ' + str(cfg.optimizer) + '\n')
    fcfg.write('-- class num: ' + str(cfg.NUM_OF_CLASSESS) + '\n')
    fcfg.write('-- total iter: ' + str(cfg.total_iter) + '\n')
    fcfg.write('-- Time acc: ' + str(cfg.is_time_acc) + '\n')
    fcfg.write('-- Acc interval: ' + str(cfg.acc_interval) + '\n')
    fcfg.write('-- Start Acc iter: ' + str(cfg.start_show_iter) + '\n')
    fcfg.write('-- Is save step: ' + str(cfg.is_save_step) + '\n')
    fcfg.write('-- Start save step: ' + str(cfg.start_save_step) + '\n')
    fcfg.write('-- save ecpoch: ' + str(cfg.save_step_inter) + '\n')
    fcfg.write('-- model save num: ' + str(cfg.model_save_num) + '\n')
    fcfg.write('-- summary interval: ' + str(cfg.summary_interval) + '\n')
    fcfg.write('-- weight decay: ' + str(cfg.weight_decay) + '\n')
    fcfg.write('-- Freeze BN: ' + str(cfg.freeze_bn) + '\n')
    fcfg.write('-- Decay rate: ' + str(cfg.decay_rate) + '\n')
    fcfg.write('-- minScale: ' + str(cfg.minScale) + '\n')
    fcfg.write('-- maxScale: ' + str(cfg.maxScale) + '\n')
    fcfg.write('-- random scale: ' + str(cfg.random_scale) + '\n')
    fcfg.write('-- random mirror: ' + str(cfg.random_mirror) + '\n')
    fcfg.write('-- random crop: ' + str(cfg.random_crop_pad) + '\n')
    fcfg.write('-- Validation on :' + str(cfg.val_data_list) + '\n')
    fcfg.close()

    last_summary_time = time.time()
    last_acc_time = time.time()
    record = train_number / cfg.batch_size      # iter number of each epoch
    if cfg.is_save_step:                        # save with step
        running_count = count
        epo = int(count / record)
    if cfg.is_save_epoch:                       # save with epoch
        running_count = int(epo * record)
        epo = count


    train_start_time = time.time()
    lossTr_list = []


    # Change the graph for read only
    sess.graph.finalize()

    while running_count < cfg.total_iter:
        time_start = time.time()
        itr = 0
        while itr < int(record):
            itr += 1
            running_count += 1

            # log last 10 model
            if running_count > (cfg.total_iter - 10) and cfg.is_save_last10_model:
                saver.save(sess, cfg.save_dir + 'model.ckpt', int(running_count))
                print('Model has been saved:' + str(running_count))

            # more than total iter, stopping training
            if running_count > cfg.total_iter:
                break

            feed_dict = {step_ph: (running_count)}

            # save summary
            now = time.time()
            if now - last_summary_time > cfg.summary_interval:
                summary_str = sess.run(summary_op, feed_dict={step_ph: running_count})
                train_writer.add_summary(summary_str, running_count)
                last_summary_time = now
                score_map, img, gt = sess.run([_mask, _img, _gt], feed_dict=feed_dict)
                img = np.array(img + cfg.IMG_MEAN, np.uint8)
                score_map = score_map * 10
                gt = gt * 10

                save_temp = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 3, 3), np.uint8)
                save_temp[0:cfg.IMAGE_HEIGHT, 0:cfg.IMAGE_WIDTH, :] = img
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH:cfg.IMAGE_WIDTH * 2, :] = gt
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 2:cfg.IMAGE_WIDTH * 3, :] = score_map
                cv2.imwrite(cfg.save_dir + 'temp_img/' + str(now) + '_mask.jpg', save_temp)

            time_s = time.time()

            # Run the zero_ops to initialize it
            sess.run(zero_ops)

            # Accumulate the gradients 'n_minibatches' times in accum_vars using accum_ops
            for i in range(cfg.Gradient_Accumulation):
                sess.run(accum_ops, feed_dict=feed_dict)
            train_loss, ls_ce, ls_l2, lr = sess.run([loss, ce_loss, l2_losses, learning_rate], feed_dict=feed_dict)
            lossTr_list.append(ls_ce)

            # Run the train_step ops to update the weights based on your accumulated gradients
            sess.run(train_step, feed_dict=feed_dict)


            time_e = time.time()

            print("Epo: %d, Step: %d, Train_loss:%g, ce: %g, l2:%g,  lr:%g, time:%g" %
                  (epo, running_count, train_loss, ls_ce, ls_l2, lr, time_e - time_s))

            # Save step model
            if cfg.is_save_step and (running_count % cfg.save_step_inter) == 0 \
                    and running_count >= cfg.start_save_step:
                saver.save(sess, cfg.save_dir + 'model.ckpt', int(running_count))
                print('Model has been saved:' + str(running_count))
                files = os.path.join(cfg.save_dir + 'model.ckpt-*.data-00000-of-00001')
                sfile = glob.glob(files)
                if len(sfile) > cfg.model_save_num:
                    steps = []
                    for s in sfile:
                        part = s.split('.')
                        re = int(part[1].split('-')[1])
                        steps.append(re)
                    re = min(steps)
                    model = cfg.save_dir + 'model.ckpt-' + str(re)
                    os.remove(model + '.data-00000-of-00001')
                    os.remove(model + '.index')
                    os.remove(model + '.meta')
                    print('Remove Model:' + model)


        epo += 1
        # Save epoch model
        if cfg.is_save_epoch and (epo % cfg.save_epoch_inter) == 0 and epo >= cfg.start_save_epoch:
            saver.save(sess, cfg.save_dir + 'model.ckpt', epo)
            print('Model has been saved:' + str(epo))
            files = os.path.join(cfg.save_dir + 'model.ckpt-*.data-00000-of-00001')
            sfile = glob.glob(files)
            if len(sfile) > cfg.model_save_num:
                steps = []
                for s in sfile:
                    part = s.split('.')
                    re = int(part[1].split('-')[1])
                    steps.append(re)
                re = min(steps)
                model = cfg.save_dir + 'model.ckpt-' + str(re)
                os.remove(model + '.data-00000-of-00001')
                os.remove(model + '.index')
                os.remove(model + '.meta')
                print('Remove Model:' + model)

        time_end = time.time()
        print('Epo ' + str(epo) + ' use time: ' + str(time_end - time_start))

    # saver.save(sess, cfg.save_dir + 'last.ckpt')    # save last model

    train_end_time = time.time()
    print('Train total use: ' + str((train_end_time-train_start_time) / 3600) + ' h')
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
