from __future__ import print_function
import numpy as np
import config as cfg


IMAGE_HEIGHT = cfg.IMAGE_HEIGHT
IMAGE_WIDTH = cfg.IMAGE_WIDTH

gr_shape = cfg.IMAGE_HEIGHT


# For sequnce input, output single frame
def data_crop_test_output(session, gr_data, logits, image, mean, std, mean_rgb, crop_size_h, crop_size_w, stride):
    image_h = image.shape[0]
    image_w = image.shape[1]
    pad_h = 0
    pad_w = 0
    if image_h >= crop_size_h and image_w >= crop_size_w:
        image_pad = image
    else:
        if image_h < crop_size_h:
            pad_h = crop_size_h - image_h
        if image_w < crop_size_w:
            pad_w = crop_size_w - image_w
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
    image_pad = np.asarray(image_pad, dtype='float32')

    mean_rgb_sub = mean_rgb
    for i in range(1, cfg.seq_num):
        mean_rgb_sub = np.concatenate([mean_rgb_sub, mean_rgb], axis=0)
    image_pad = image_pad - mean_rgb_sub  # sub rgb mean

    image_crop_batch = []
    x_start = [x for x in range(0, image_pad.shape[0] - crop_size_h + 1, stride)]
    y_start = [y for y in range(0, image_pad.shape[1] - crop_size_w + 1, stride)]
    if (image_pad.shape[0] - crop_size_h) % stride != 0:
        x_start.append(image_pad.shape[0] - crop_size_h)
    if (image_pad.shape[1] - crop_size_w) % stride != 0:
        y_start.append(image_pad.shape[1] - crop_size_w)
    for x in x_start:
        for y in y_start:
            image_crop_batch.append(image_pad[x:x + crop_size_h, y:y + crop_size_w])

    logit = []
    for crop_batch in image_crop_batch:
        lo = session.run(logits, feed_dict={gr_data: [crop_batch]})
        logit.append(lo[0])

    num_class = cfg.NUM_OF_CLASSESS
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
            crop_logits = logit[crop_index]
            score_map[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += crop_logits
            count[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += 1
            crop_index += 1

    score_map = score_map[:image_h, :image_w] / count[:image_h, :image_w]
    return score_map

# For pose, depth and intrinsics to input
def data_crop_test_output_pose(session, image_data, depth_data, pose_data, ins_data, logits, image, depth, pose, ins,
                               mean, std, mean_rgb, crop_size_h, crop_size_w, stride):
    image_h = image.shape[0]
    image_w = image.shape[1]
    pad_h = 0
    pad_w = 0
    if image_h >= crop_size_h and image_w >= crop_size_w:
        image_pad = image
        depth_pad = depth
    else:
        if image_h < crop_size_h:
            pad_h = crop_size_h - image_h
        if image_w < crop_size_w:
            pad_w = crop_size_w - image_w
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
        depth_pad = np.pad(depth, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
    image_pad = np.asarray(image_pad, dtype='float32')
    depth_pad = np.asarray(depth_pad, dtype='float32')

    mean_rgb_sub = mean_rgb
    for i in range(1, cfg.seq_num):
        mean_rgb_sub = np.concatenate([mean_rgb_sub, mean_rgb], axis=0)
    image_pad = image_pad - mean_rgb_sub  # sub rgb mean

    image_crop_batch = []
    depth_crop_batch = []
    x_start = [x for x in range(0, image_pad.shape[0] - crop_size_h + 1, stride)]
    y_start = [y for y in range(0, image_pad.shape[1] - crop_size_w + 1, stride)]
    if (image_pad.shape[0] - crop_size_h) % stride != 0:
        x_start.append(image_pad.shape[0] - crop_size_h)
    if (image_pad.shape[1] - crop_size_w) % stride != 0:
        y_start.append(image_pad.shape[1] - crop_size_w)
    for x in x_start:
        for y in y_start:
            image_crop_batch.append(image_pad[x:x + crop_size_h, y:y + crop_size_w])
            depth_crop_batch.append(depth_pad[x:x + crop_size_h, y:y + crop_size_w])

    logit = []
    for i, crop_batch in enumerate(image_crop_batch):
        lo = session.run(logits, feed_dict={image_data: [crop_batch], depth_data: [depth_crop_batch[i]],
                                            pose_data: [pose], ins_data: [ins]})
        logit.append(lo[0])

    num_class = cfg.NUM_OF_CLASSESS
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
            crop_logits = logit[crop_index]
            score_map[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += crop_logits
            count[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += 1
            crop_index += 1

    score_map = score_map[:image_h, :image_w] / count[:image_h, :image_w]
    return score_map

# For multiple frames to output
def data_crop_test_output_seq(session, gr_data, logits, image, mean, std, mean_rgb, crop_size_h, crop_size_w, stride):
    image_h = image.shape[0]
    image_w = image.shape[1]
    pad_h = 0
    pad_w = 0
    if image_h >= crop_size_h and image_w >= crop_size_w:
        image_pad = image
    else:
        if image_h < crop_size_h:
            pad_h = crop_size_h - image_h
        if image_w < crop_size_w:
            pad_w = crop_size_w - image_w
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
    image_pad = np.asarray(image_pad, dtype='float32')

    mean_rgb_sub = mean_rgb
    for i in range(1, cfg.seq_num):
        mean_rgb_sub = np.concatenate([mean_rgb_sub, mean_rgb], axis=0)
    image_pad = image_pad - mean_rgb_sub  # sub rgb mean

    image_crop_batch = []
    x_start = [x for x in range(0, image_pad.shape[0] - crop_size_h + 1, stride)]
    y_start = [y for y in range(0, image_pad.shape[1] - crop_size_w + 1, stride)]
    if (image_pad.shape[0] - crop_size_h) % stride != 0:
        x_start.append(image_pad.shape[0] - crop_size_h)
    if (image_pad.shape[1] - crop_size_w) % stride != 0:
        y_start.append(image_pad.shape[1] - crop_size_w)
    for x in x_start:
        for y in y_start:
            image_crop_batch.append(image_pad[x:x + crop_size_h, y:y + crop_size_w])

    logit = []
    for crop_batch in image_crop_batch:
        lo = session.run(logits, feed_dict={gr_data: [crop_batch]})
        logit_img = lo[0][0]
        for i in range(1, cfg.seq_num):
            logit_img = np.concatenate([logit_img, lo[i][0]], axis=2)
        logit.append(logit_img)


    num_class = int(cfg.NUM_OF_CLASSESS * cfg.seq_num)
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
            crop_logits = logit[crop_index]
            score_map[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += crop_logits
            count[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += 1
            crop_index += 1

    score_map = score_map[:image_h, :image_w] / count[:image_h, :image_w]
    return score_map

