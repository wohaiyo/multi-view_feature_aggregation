import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
import config as cfg
import cv2
from scipy import misc
import random

def image_resizing(imgs, label, depths):
    '''
    Random resize the images and labels between 0.5 to 1.5 for height or width
    :param img:
    :param label:
    :return: img and label
    '''

    scale = tf.cast(np.random.uniform(0.75, 1.25), dtype=tf.float32)
    img_h = tf.shape(img)[0]
    img_w = tf.shape(img)[1]
    h_scale = tf.to_int32(tf.to_float(img_h) * scale)
    w_scale = tf.to_int32(tf.to_float(img_w) * scale)

    if np.random.uniform(0, 1) < 0.5:
        h_new = h_scale
        w_new = img_w
    else:
        h_new = img_h
        w_new = w_scale

    new_shape = tf.stack([h_new, w_new])
    img_d = tf.image.resize_images(img, new_shape)
    label_d = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label_d = tf.squeeze(label_d, squeeze_dims=[0])

    return img_d, label_d


def image_scaling(imgs, label, depths, intrinsics):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
      mask: 3 layer(top, mid, bot) mask to scale.
      boundary: boundary mask to scale.
      scale: ECP: [0.38, 1.0] [0.58, 1.25] [0.75, 1.5]
             eTRIMS:[0.33, 0.75] [0.5, 1.0] [0.66, 1.25]
    """

    # # fixed scales: no useless because the scale is fixed at one value
    # scales = [0.75, 0.87, 1.0, 1.15, 1.3, 1.45, 1.6, 1.75]
    # sc = random.sample(scales, 1)
    # print(sc)
    # scale = tf.convert_to_tensor(sc, dtype=tf.float32)

    # random scales range(0.75, 1.75)
    scale = tf.random_uniform([1], minval=cfg.minScale, maxval=cfg.maxScale, dtype=tf.float32, seed=None)

    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(imgs)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(imgs)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    imgs = tf.image.resize_images(imgs, new_shape)

    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    depths =tf.image.resize_images(depths, new_shape)

    intrinsics = intrinsics * scale
    r3 = tf.constant([0., 0., 1.], shape=[1, 1, 3])
    r3 = tf.tile(r3, [cfg.seq_num, 1, 1])
    intrinsics = tf.concat([intrinsics[:, 0:2, :], r3], axis=1)

    return imgs, label, depths, intrinsics


def image_mirroring(imgs, label, depths):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
      mask: 3 layer mask to mirror.
      boundary: boundary mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    imgs = tf.reverse(imgs, mirror)
    label = tf.reverse(label, mirror)
    depths = tf.reverse(depths, mirror)

    return imgs, label, depths


def random_crop_and_pad_image_label_depth(images, label, depths, crop_h, crop_w, ignore_label=255):   # img:[h, w, 3*n] label:[h,w,n]
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      mask: 3 layer mask to crop/pad.
      boundary: boundary mask to crop/pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.

    depths = tf.cast(depths, dtype=tf.float32)

    combined = tf.concat(axis=2, values=[images, label, depths])
    image_shape = tf.shape(images)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),     # top-left
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = images.get_shape()[-1]

    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, int(5 * cfg.seq_num)])
    img_crop = combined_crop[:, :, :last_image_dim]

    label_crop = combined_crop[:, :, last_image_dim:last_image_dim + cfg.seq_num]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    depth_crop = combined_crop[:, :, (last_image_dim + cfg.seq_num):(last_image_dim + cfg.seq_num + cfg.seq_num)]

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, last_image_dim))
    label_crop.set_shape((crop_h, crop_w, cfg.seq_num))
    depth_crop.set_shape((crop_h, crop_w, cfg.seq_num))

    return img_crop, label_crop, depth_crop


def get_image_and_labels(images, label, depths, crop_h, crop_w):
    # Set static shape so that tensorflow knows shape at compile time.

    # # For other 512 x 512
    # new_shape = tf.squeeze(tf.stack([crop_h, crop_w]))
    # image = tf.image.resize_images(image, new_shape)
    # label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    # label = tf.squeeze(label, squeeze_dims=[0])

    images.set_shape((crop_h, crop_w, int(3*cfg.seq_num)))
    label.set_shape((crop_h, crop_w, cfg.seq_num))
    depths.set_shape((crop_h, crop_w, cfg.seq_num))

    return images, label, depths


def random_brightness_contrast_hue_satu(img):
    '''
    Random birght and countrast
    :param img:
    :return:
    '''
    if np.random.uniform(0, 1) < 0.5:
        distorted_image = tf.image.random_brightness(img, max_delta=32./255.)
        distorted_image = tf.image.random_saturation(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
    else:
        distorted_image = tf.image.random_brightness(img, max_delta=32./255.)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_saturation(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)

    image = distorted_image

    return image


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/label
                                                        /path/to/mask  /path/to/boundary '.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    labels = []
    depths = []
    poses  = []
    intrinsicses = []

    for line in f:
        try:
            image, label, depth, pose, intrinsics  = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = label = depth = pose = intrinsics = line.strip("\n")
        images.append(data_dir + image)
        labels.append(data_dir + label)
        depths.append(data_dir + depth)
        poses.append(data_dir + pose)
        intrinsicses.append(data_dir + intrinsics)


    return images, labels, depths, poses, intrinsicses


def block_patch(img_stack, margin=50):
    new_stack = []
    k = int(cfg.seq_num / 2)

    for i in range(cfg.seq_num):
        img = img_stack[:, :, int(i * 3):int((i + 1) * 3)]

        if i == k:
        # if i != k:
        # prob = np.random.uniform(0, 1)
        # if i != k and prob < 0.5:
            shape = img.get_shape().as_list()

            # create patch in random size
            pad_size = tf.random_uniform([2], minval=int(shape[1] / 5), maxval=int(shape[1] / 3), dtype=tf.int32)
            patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)

            h_ = tf.random_uniform([1], minval=margin, maxval=shape[0] - pad_size[0] - margin, dtype=tf.int32)[0]
            w_ = tf.random_uniform([1], minval=margin, maxval=shape[1] - pad_size[1] - margin, dtype=tf.int32)[0]

            padding = [[h_, shape[0] - h_ - pad_size[0]], [w_, shape[1] - w_ - pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

            img = tf.multiply(img, padded)  # inpainted region is 0

        new_stack.append(img)

    return tf.concat(new_stack, axis=2)

def block_img_label(img_stack, label_stack, margin=10):
    new_img_stack = []
    new_label_stack = []
    k = int(cfg.seq_num / 2)

    for i in range(cfg.seq_num):
        img = img_stack[:, :, int(i * 3):int((i + 1) * 3)]
        label = tf.cast(label_stack[:, :, i:i + 1], tf.float32)

        # if i == 10:
        # if i != k:
        prob = np.random.uniform(0, 1)
        if prob < 1.0:
        # if i != k and prob < 0.5:
            shape = img.get_shape().as_list()
            shape_label = label.get_shape().as_list()

            # create patch in random size
            pad_size = tf.random_uniform([2], minval=int(shape[1] / 5), maxval=int(shape[1] / 3), dtype=tf.int32) # [1/5, 1/3]
            patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)
            patch_label = tf.zeros([pad_size[0], pad_size[1], shape_label[-1]], dtype=tf.float32)

            # Control the locations of block (height)
            prob2 = np.random.uniform(0,1)
            if prob2 < 0:
                h_margin = np.random.randint(int(shape[0] / 2), int(shape[0] / 3 * 2 - margin))
                # print(i, h_margin)
                h_ = tf.random_uniform([1], minval=h_margin, maxval=shape[0] - pad_size[0], dtype=tf.int32)[0]
            else:
                h_ = tf.random_uniform([1], minval=margin, maxval=shape[0] - pad_size[0] - margin, dtype=tf.int32)[0]
            w_ = tf.random_uniform([1], minval=margin, maxval=shape[1] - pad_size[1] - margin, dtype=tf.int32)[0]

            padding = [[h_, shape[0] - h_ - pad_size[0]], [w_, shape[1] - w_ - pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
            padded_label = tf.pad(patch_label, padding, "CONSTANT", constant_values=1)

            img = tf.multiply(img, padded)                  # inpainted region is 0
            label = tf.multiply(label, padded_label)        # inpainted region is 0
            label.set_shape((shape_label[0], shape_label[1], 1))

        new_img_stack.append(img)
        new_label_stack.append(tf.cast(label, tf.uint8))

    return tf.concat(new_img_stack, axis=2), tf.concat(new_label_stack, axis=2)

def block_img_label_mul(img_stack, label_stack, margin=10):
    new_img_stack = []
    new_label_stack = []
    k = int(cfg.seq_num / 2)

    for m in range(2):
        for i in range(cfg.seq_num):
            img = img_stack[:, :, int(i * 3):int((i + 1) * 3)]
            label = tf.cast(label_stack[:, :, i:i + 1], tf.float32)

            prob = np.random.uniform(0, 1)
            if prob < 1.0:
                shape = img.get_shape().as_list()
                shape_label = label.get_shape().as_list()

                # create patch in random size
                # pad_size = [shape[1] // 5, shape[1] // 5]
                pad_size = tf.random_uniform([2], minval=int(shape[1] / 5), maxval=int(shape[1] / 3),
                                             dtype=tf.int32)  # [1/5, 1/3]
                patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)
                patch_label = tf.zeros([pad_size[0], pad_size[1], shape_label[-1]], dtype=tf.float32)

                h_ = tf.random_uniform([1], minval=margin, maxval=shape[0] - pad_size[0] - margin, dtype=tf.int32)[0]
                w_ = tf.random_uniform([1], minval=margin, maxval=shape[1] - pad_size[1] - margin, dtype=tf.int32)[0]

                padding = [[h_, shape[0] - h_ - pad_size[0]], [w_, shape[1] - w_ - pad_size[1]], [0, 0]]
                padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
                padded_label = tf.pad(patch_label, padding, "CONSTANT", constant_values=1)

                img = tf.multiply(img, padded)                  # inpainted region is 0
                label = tf.multiply(label, padded_label)        # inpainted region is 0
                label.set_shape((shape_label[0], shape_label[1], 1))

            new_img_stack.append(img)
            new_label_stack.append(tf.cast(label, tf.uint8))

        img_stack = tf.concat(new_img_stack, axis=2)
        label_stack = tf.concat(new_label_stack, axis=2)
        new_img_stack = []
        new_label_stack = []

    return img_stack, label_stack

def block_img_label_fix_size(img_stack, label_stack, margin=10):
    new_img_stack = []
    new_label_stack = []
    new_occ_stack = []
    k = int(cfg.seq_num / 2)

    for i in range(cfg.seq_num):
        img = img_stack[:, :, int(i * 3):int((i + 1) * 3)]
        label = tf.cast(label_stack[:, :, i:i + 1], tf.float32)


        # if i != k:
        prob = np.random.uniform(0, 1)
        if prob < 1.0:
        # if i == k:
            shape = img.get_shape().as_list()
            shape_label = label.get_shape().as_list()

            # create patch in random size
            # pad_size = tf.random_uniform([2], minval=int(shape[1] / 5), maxval=int(shape[1] / 3),
            #                                  dtype=tf.int32) # [1/5, 1/3]
            # pad_size = [0, 0] # 0.0
            # pad_size = [74, 74] # 0.025
            # pad_size = [104, 104] # 0.05
            # pad_size = [117, 117] # 0.063
            pad_size = [128, 128] # 0.075
            # pad_size = [148, 148] # 0.1
            # pad_size = [165, 165] # 0.125
            # pad_size = [181, 181] # 0.15
            # pad_size = [208, 208] # 0.2
            patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)
            patch_label = tf.zeros([pad_size[0], pad_size[1], shape_label[-1]], dtype=tf.float32)

            h_ = tf.random_uniform([1], minval=margin, maxval=shape[0] - pad_size[0] - margin, dtype=tf.int32)[0]
            w_ = tf.random_uniform([1], minval=margin, maxval=shape[1] - pad_size[1] - margin, dtype=tf.int32)[0]

            padding = [[h_, shape[0] - h_ - pad_size[0]], [w_, shape[1] - w_ - pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
            padded_label = tf.pad(patch_label, padding, "CONSTANT", constant_values=1)

            img = tf.multiply(img, padded)                  # inpainted region is 0
            label = tf.multiply(label, padded_label)        # inpainted region is 0
            label.set_shape((shape_label[0], shape_label[1], 1))
            padded_label.set_shape((shape_label[0], shape_label[1], 1))

        new_img_stack.append(img)
        new_label_stack.append(tf.cast(label, tf.uint8))
        new_occ_stack.append(tf.cast(padded_label, tf.uint8))

    return tf.concat(new_img_stack, axis=2), tf.concat(new_label_stack, axis=2), tf.concat(new_occ_stack, axis=2)

def block_img_label_fix_size_loc(img_stack, label_stack, margin=10):
    new_img_stack = []
    new_label_stack = []
    k = int(cfg.seq_num / 2)

    for i in range(cfg.seq_num):
        img = img_stack[:, :, int(i * 3):int((i + 1) * 3)]
        label = tf.cast(label_stack[:, :, i:i + 1], tf.float32)

        # if i == 10:
        # if i != k:
        prob = np.random.uniform(0, 1)
        if prob < 1.0:
        # if i != k and prob < 0.5:
            shape = img.get_shape().as_list()
            shape_label = label.get_shape().as_list()

            # create patch in random size
            pad_size = [shape[0] // 4, shape[1] // 4]
            patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)
            patch_label = tf.zeros([pad_size[0], pad_size[1], shape_label[-1]], dtype=tf.float32)

            h_ = random.choice([margin, shape[0]//4, shape[0]//2, shape[0]-margin-pad_size[0]])
            w_ = random.choice([margin, shape[1]//4, shape[1]//2, shape[1]-margin-pad_size[1]])

            padding = [[h_, shape[0] - h_ - pad_size[0]], [w_, shape[1] - w_ - pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
            padded_label = tf.pad(patch_label, padding, "CONSTANT", constant_values=1)

            img = tf.multiply(img, padded)                  # inpainted region is 0
            label = tf.multiply(label, padded_label)        # inpainted region is 0
            label.set_shape((shape_label[0], shape_label[1], 1))

        new_img_stack.append(img)
        new_label_stack.append(tf.cast(label, tf.uint8))

    return tf.concat(new_img_stack, axis=2), tf.concat(new_label_stack, axis=2)

def read_images_from_disk(image_queue, label_queue, depth_queue, intrinsics, input_size, random_scale, random_resize, random_mirror, random_color, random_crop_pad,
                          ignore_label, img_mean):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      random_color: random brightness, contrast, hue and saturation.
      random_crop_pad: random crop and padding for h and w of image
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """
    h, w = input_size

    img_reader = tf.WholeFileReader()
    _, img_contents = img_reader.read(image_queue)
    label_reader = tf.WholeFileReader()
    _, label_contents = label_reader.read(label_queue)
    depth_reader = tf.WholeFileReader()
    _, depth_contents = depth_reader.read(depth_queue)

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    image_seq = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)      # B G R

    label_seq = tf.image.decode_png(label_contents, channels=1)

    depth_seq = tf.image.decode_png(depth_contents, channels=1, dtype=tf.uint16)
    depth_seq = tf.cast(depth_seq, tf.float32) / 1000

    ori_w = cfg.IMAGE_ORI_WIDTH
    # stack image sequence
    img_stack = tf.slice(image_seq, [0, 0, 0], [-1, ori_w, -1])
    label_stack = tf.slice(label_seq, [0, 0, 0], [-1, ori_w, -1])
    depth_stack = tf.slice(depth_seq, [0, 0, 0], [-1, ori_w, -1])


    for i in range(1, cfg.seq_num):
        img = tf.slice(image_seq, [0, i * ori_w, 0], [-1, ori_w, -1])
        label = tf.slice(label_seq, [0, i * ori_w, 0], [-1, ori_w, -1])
        depth = tf.slice(depth_seq, [0, i * ori_w, 0], [-1, ori_w, -1])
        img_stack = tf.concat([img_stack, img], axis=2)
        label_stack = tf.concat([label_stack, label], axis=2)
        depth_stack = tf.concat([depth_stack, depth], axis=2)

    # Randomly scale the images, labels and depth
    if random_scale:
        img_stack, label_stack, depth_stack, intrinsics = image_scaling(img_stack, label_stack, depth_stack, intrinsics)

    # if random_resize:
    #     img_stack, label, depth_stack = image_resizing(img_stack, label, depth_stack)

    # Randomly mirror the images and labels.
    if random_mirror:
        img_stack, label_stack, depth_stack = image_mirroring(img_stack, label_stack, depth_stack)

    # # random_brightness_contrast_hue_satu
    # if random_color:
    #     img_stack = random_brightness_contrast_hue_satu(img_stack)

    # Randomly crops the images and labels.
    if random_crop_pad:
        img_stack, label_stack, depth_stack = random_crop_and_pad_image_label_depth(img_stack, label_stack, depth_stack, h, w, ignore_label)
    else:
        img_stack, label_stack, depth_stack = get_image_and_labels(img_stack, label_stack, depth_stack, h, w)

    # # Cover some regions
    # img_stack = block_patch(img_stack)
    # img_stack, label_stack_blocked = block_img_label(img_stack, label_stack)
    img_stack, label_stack_blocked, occ_stack = block_img_label_fix_size(img_stack, label_stack)
    # img_stack, label_stack_blocked = block_img_label_fix_size_loc(img_stack, label_stack)
    # img_stack, label_stack_blocked = block_img_label_mul(img_stack, label_stack)
    # label_stack_blocked = label_stack

    img_mean_stack = img_mean
    for i in range(1, cfg.seq_num):
        img_mean_stack = tf.concat([img_mean_stack, img_mean], axis=0)
    # Extract mean.
    img_stack -= img_mean_stack

    return img_stack, label_stack_blocked, occ_stack, label_stack, depth_stack, intrinsics

def read_pose_intrinsics(pose_queue, intrinsics_queue):
    '''
    Return absolute pose and camera intrinsics
    :param pose_queue:
    :param intrinsics_queue:
    :return:
    '''

    # Load absolute pose
    pose_reader = tf.TextLineReader()
    _, raw_pose_contents = pose_reader.read(pose_queue)
    rec_def = []
    for i in range(int(16 * cfg.seq_num)):
        rec_def.append([1.])
    raw_pose_vec = tf.decode_csv(raw_pose_contents,
                                record_defaults=rec_def)
    raw_pose_vec = tf.stack(raw_pose_vec)
    pose = tf.reshape(raw_pose_vec, [cfg.seq_num, 4, 4])

    # Load camera intrinsics
    cam_reader = tf.TextLineReader()
    _, raw_cam_contents = cam_reader.read(intrinsics_queue)
    rec_def = []
    for i in range(int(9 * cfg.seq_num)):
        rec_def.append([1.])
    raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                record_defaults=rec_def)
    raw_cam_vec = tf.stack(raw_cam_vec)
    intrinsics = tf.reshape(raw_cam_vec, [cfg.seq_num, 3, 3])

    # pose = tf.zeros((4, 4), dtype=tf.float32)
    # intrinsics = tf.zeros((3, 3), dtype=tf.float32)

    return pose, intrinsics

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size,
                 random_scale, random_resize, random_mirror, random_color, random_crop_pad, ignore_label, img_mean, coord):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          random_color: whether to randomly brightness, contrast, hue and satr.
          random_crop_pad: whether to randomly corp and pading images.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list, self.depth_list, self.pose_list, self.intrinsics_list = \
            read_labeled_image_list(self.data_dir, self.data_list)

        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.depths = tf.convert_to_tensor(self.depth_list, dtype=tf.string)
        self.poses = tf.convert_to_tensor(self.pose_list, dtype=tf.string)
        self.intrinsicses = tf.convert_to_tensor(self.intrinsics_list, dtype=tf.string)

        seed = random.randint(0, 2 ** 31 - 1)
        # self.queue = tf.train.slice_input_producer([self.images, self.labels, self.depths, self.poses, self.intrinsicses],
        #                                            shuffle=True, seed=seed)  # not shuffling if it is val #
        # True: not equal, False: pre-processing data list. Default: False

        self.image_queue = tf.train.string_input_producer(self.images, shuffle=True, seed=seed)
        self.label_queue = tf.train.string_input_producer(self.labels, shuffle=True, seed=seed)
        self.depth_queue = tf.train.string_input_producer(self.depths, shuffle=True, seed=seed)
        self.pose_queue = tf.train.string_input_producer(self.poses, shuffle=True, seed=seed)
        self.intrinsics_queue = tf.train.string_input_producer(self.intrinsicses, shuffle=True, seed=seed)

        self.pose, self.intrinsics = read_pose_intrinsics(self.pose_queue, self.intrinsics_queue)

        self.image, self.label_blocked, self.occ, self.label, self.depth, self.intrinsics = read_images_from_disk(self.image_queue, self.label_queue,
                                                                   self.depth_queue, self.intrinsics, self.input_size,
                                                                   random_scale, random_resize, random_mirror,
                                                                   random_color, random_crop_pad,
                                                                   ignore_label, img_mean)



    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns: imag: float32, label: int32, depth: float, pose: float, intrinsics: float
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_block_batch, occ_batch, label_batch = tf.train.batch(
            [self.image, self.label_blocked, self.occ, self.label],
            num_elements)
        return image_batch, tf.cast(label_block_batch, dtype=tf.int32),  tf.cast(occ_batch, dtype=tf.int32), tf.cast(label_batch, dtype=tf.int32)

    def getqueue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_queue = tf.train.batch(
            [self.queue],
            num_elements)
        return image_queue


if __name__ == '__main__':

    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
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
        image_batch, label_block_batch, occ_batch, label_batch = reader.dequeue(cfg.batch_size)
        # label_block_batch = occ_batch
        # label_block_batch = label_batch - label_block_batch
        # ques = reader.getqueue(cfg.batch_size)

    with tf.Session() as se:
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=se)

        imgs, labels_block, labels = se.run([image_batch, label_block_batch, label_batch])

        img = imgs[0][:, :, 0:3]
        label_block = labels_block[0][:, :, 0:1]
        label = labels[0][:, :, 0:1]

        for i in range(1, cfg.seq_num):
            img = np.concatenate([img, imgs[0][:,:,int(3*i):int((i+1) * 3)]], axis=1)
            label_block = np.concatenate([label_block, labels_block[0][:, :, int(i):int(i + 1)]], axis=1)
            label = np.concatenate([label, labels[0][:, :, int(i):int(i + 1)]], axis=1)


        img = img + cfg.IMG_MEAN
        img = np.array(img, np.uint8)
        label_block = np.squeeze(label_block, axis=2) * 20
        label = np.squeeze(label, axis=2) * 20


        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('3_img5.png', img)
        cv2.imwrite('3_label_block.png', label_block)
        cv2.imwrite('3_label.png', label)

        coord.request_stop()
        coord.join(threads)
    print('Done')