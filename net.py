from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import config as cfg
from tensorflow.python.ops import nn
import math
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from resnet import resnet_v1

def atrous_spp16(input_feature, depth=256):      # c: 256
    '''
    aspp for deeplabv3, output_stride=16, [6, 12, 18]; output_stide=8, rate:[12, 24, 36]
    :param input_feature:
    :param k: kernel size: 1xk, kx1
    :return: feature
    '''

    # 1x1 conv
    at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

    # rate = 6
    at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=6, activation_fn=None)

    # rate = 12
    at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=12, activation_fn=None)

    # rate = 18
    at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=18, activation_fn=None)

    # image pooling
    img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
    img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
    img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                         input_feature.get_shape().as_list()[2]))

    net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                    axis=3, name='atrous_concat')
    net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

    return net

def inference_multiview_feature_aggregation(image, is_training):    # image: [h, w, 9]

    def deeplabv3_plus(image):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            with slim.arg_scope(
                    resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                         global_pool=False, output_stride=16,
                                                         spatial_squeeze=False)
            # ASPP
            aspp = atrous_spp16(net)
            with tf.variable_scope('decoder'):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                net = tf.image.resize_images(aspp, low_level_features_shape)
                net = tf.concat([net, low_level_features], axis=3)
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

        return net

    def cosine(q, a):  # cosine similarity
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 3))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 3))
        pooled_mul_12 = tf.reduce_sum(q * a, 3)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
        # score = standard_norm(score)    # norm
        return score

    def calculate_dissimilar_map(net_key, net_feats):
        dissimilar_maps = []
        for feat in net_feats:
            similar_map = cosine(net_key, feat)
            dissimilar_map = 1 - similar_map
            dissimilar_maps.append(dissimilar_map)
        return dissimilar_maps

    def fusion_module(cur_view, other_views):
        with tf.variable_scope("fusion_module", reuse=tf.AUTO_REUSE):
            b, h, w, c = cur_view.get_shape().as_list()
            other_views.append(cur_view)
            feat_concat = tf.concat(other_views, axis=3)
            feat_selection = slim.conv2d(feat_concat, c, [1, 1], scope='conv1x1')
            feat_conv1 = slim.conv2d(feat_selection, c, [3, 3], scope='conv3x3_1')
            feat_conv2 = slim.conv2d(feat_conv1, c, [3, 3], scope='conv3x3_2')
        return feat_conv2

    def fusion_weight(cur_view, other_views):
        b, h, w, c = cur_view.get_shape().as_list()
        other_views.append(cur_view)
        view_concat = tf.concat(other_views, axis=3)
        with tf.variable_scope("weighted_module", reuse=tf.AUTO_REUSE):

            weight_map = slim.conv2d(view_concat, cfg.seq_num, [1, 1], scope='weight_map', activation_fn=None)

            # view_down = slim.conv2d(view_concat, c, [1, 1], scope='conv_down')
            # view_refine1 = slim.conv2d(view_down, c, [3, 3], scope='conv_refine1')
            # view_refine2 = slim.conv2d(view_refine1, c, [3, 3], scope='conv_refine2')
            # weight_map = slim.conv2d(view_refine2, cfg.seq_num, [1, 1], scope='weight_map', activation_fn=None)

        weight_map = tf.nn.softmax(weight_map)

        view_concat_w = []
        for i, feat_view in enumerate(other_views):
            view_concat_w.append(feat_view * weight_map[:, :, :, i:i+1])

        cur_view_w = view_concat_w[-1]
        del view_concat_w[-1]
        return cur_view_w, view_concat_w, weight_map

    k = int(cfg.seq_num / 2)
    img_shape = image.get_shape().as_list()
    net_feats = []

    # Feature extraction
    for i in range(cfg.seq_num):
        image_in = image[:, :, :, int(i * 3):int((i + 1) * 3)]
        # net_near = denseASPPNet(image_in)
        net_near = deeplabv3_plus(image_in)
        # net_near = danet(image_in)
        # net_near = pspnet(image_in)
        net_feats.append(net_near)

    other_view = net_feats.copy()
    key_feat = other_view[k]
    del other_view[k]
    simi_maps1 = calculate_dissimilar_map(key_feat, other_view)

    # Fuse key view with other views
    attention_feats = []
    refined_feats = []
    wmaps = []
    for i in range(len(net_feats)):
        net_feats_temp = net_feats.copy()
        key_feat = net_feats_temp[i]
        del net_feats_temp[i]

        # # only fusion
        # refined_feat = fusion_module2(key_feat, net_feats_temp)
        # refined_feats.append(refined_feat)
        # weight_map = refined_feat[:, :, :, 0:cfg.NUM_OF_CLASSESS]
        # wmaps.append(weight_map)

        # Aggregation and fusion
        cur_view_w, view_concat_w, weight_map = fusion_weight(key_feat, net_feats_temp)
        # cur_view_w, view_concat_w, weight_map = fusion_weight_refine(key_feat, net_feats_temp)
        # cur_view_w = key_feat
        # view_concat_w = net_feats_temp
        # weight_map = key_feat[:, :, :, 0:cfg.seq_num]

        att_logits = []
        for seq in range(cfg.seq_num):
            if seq < i:
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    logit = slim.conv2d(view_concat_w[seq], cfg.NUM_OF_CLASSESS, [1, 1], scope='logits',
                                        trainable=is_training,
                                        activation_fn=None, normalizer_fn=None)
            elif seq == i:
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    logit = slim.conv2d(cur_view_w, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                                        activation_fn=None, normalizer_fn=None)
            else:
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    logit = slim.conv2d(view_concat_w[seq - 1], cfg.NUM_OF_CLASSESS, [1, 1], scope='logits',
                                        trainable=is_training,
                                        activation_fn=None, normalizer_fn=None)
            logit = tf.image.resize_images(logit, [img_shape[1], img_shape[2]])
            att_logits.append(logit)


        attention_feats.append(att_logits)
        refined_feat = fusion_module(cur_view_w, view_concat_w)
        refined_feats.append(refined_feat)
        wmaps.append(weight_map)


    # Similar maps 2
    other_view = refined_feats.copy()
    key_feat = other_view[k]
    del other_view[k]
    simi_maps2 = calculate_dissimilar_map(key_feat, other_view)

    # Similar loss
    similar_loss_maps = []
    for i in range(len(refined_feats)):
        for j in range(i+1, len(refined_feats)):
            dis_map = (1 - cosine(refined_feats[i], refined_feats[j]))
            similar_loss_maps.append(dis_map)

    # Classification for views after fusion
    net_cls = []
    for feat in refined_feats:
        with tf.variable_scope("fusion", reuse=tf.AUTO_REUSE):
            logits_refine = slim.conv2d(feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_refine', trainable=is_training,
                                        activation_fn=None, normalizer_fn=None)
        logits_refine = tf.image.resize_images(logits_refine, [img_shape[1], img_shape[2]])
        net_cls.append(logits_refine)
    label_pred = tf.expand_dims(tf.argmax(net_cls[k], axis=3, name="prediction"), dim=3)

    if is_training:
        return label_pred, attention_feats, net_cls, simi_maps1, simi_maps2, similar_loss_maps
    else:
        return label_pred, net_cls[k], net_cls, simi_maps1, simi_maps2, similar_loss_maps, wmaps
