from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob

import imageio
import config as cfg
import time
import csv
from tensorflow.python import pywrap_tensorflow

from net import inference_multiview_feature_aggregation

from utils import pred_vision, eval_img2, eval_fscore
from pre_data2 import data_crop_test_output
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_gpu

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "multi_eval", "Mode train/ test/ visualize")    # visualize  train mul_eval


IMAGE_SIZE = None
class_names_ignore_background = []                                      # Ignore backgournd label
for i in range(1, len(cfg.class_names)):
    class_names_ignore_background.append(cfg.class_names[i])
cfg.class_names = class_names_ignore_background


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
    _scope = 'vgg_16'
    _variables_to_fix = []

    for v in variables:

        if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)

    return variables_to_restore

def fast_hist(a, b, n):         # a: gt, b: pred
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, int(cfg.seq_num*3)], name="input_image")


    # aggregation and fusion
    _, fc8s_logits, _, attention, attention2, _, wmaps = inference_multiview_feature_aggregation(
        image, is_training=False)


    f = open(cfg.test_data_list, 'r')
    img_list = []
    label_list = []
    depth_list = []
    pose_list = []
    intrinsics_list = []
    for line in f:
        try:
            image_name, label, depth, pose, intrinsics = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image_name = label = depth = pose = intrinsics = line.strip("\n")
        img_list.append(cfg.test_data_dir + image_name)
        label_list.append(cfg.test_data_dir + label)
        depth_list.append(cfg.test_data_dir + depth)
        pose_list.append(cfg.test_data_dir + pose)
        intrinsics_list.append(cfg.test_data_dir + intrinsics)


    logits = tf.nn.softmax(fc8s_logits)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()


    is_best = False          # Use the best model to evaluate

    files = os.path.join(cfg.save_dir + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) >= 0:
        sess.run(tf.global_variables_initializer())
        if is_best:
            model = cfg.save_dir + 'best.ckpt'
            # model = cfg.save_dir + 'last.ckpt'
        else:
            sfile = glob.glob(files)
            steps = []
            for s in sfile:
                part = s.split('.')
                step = int(part[1].split('-')[1])
                steps.append(step)
            epo = max(steps)


            # Which model to eval
            model = cfg.save_dir + 'model.ckpt-' + str(epo)

        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')
    else:
        # restore from pre-train on imagenet or pre-trained
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))
        if os.path.exists(cfg.pre_trained_model) or os.path.exists(cfg.pre_trained_model + '.index'):
            var_keep_dic = get_variables_in_checkpoint_file(cfg.pre_trained_model)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            # var_to_restore = [val for val in variables if 'conv1' in val.name or 'conv2' in val.name or
            #                   'conv3' in val.name or 'conv4' in val.name or 'conv5' in val.name]
            if len(variables_to_restore) > 0:
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, cfg.pre_trained_model)
                print('Vgg model pre-train Loaded')
            else:
                print('Model inited random.')
        else:
            print('Model inited random.')

    if FLAGS.mode == "train":
        print('Start train ...')

    elif FLAGS.mode == 'multi_eval':
        print('---------Start multi-scale eval-------------')
        crop_size_h = cfg.IMAGE_HEIGHT # 480 512 500 224
        crop_size_w = cfg.IMAGE_WIDTH
        print('crop size: ' + str(crop_size_h))
        stride = int(crop_size_h / 3)
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]
        # mean_rgb = [123.68, 116.78, 103.94]  # rgb mean subtract

        mean_bgr = [103.94, 116.78, 123.68]


        if not os.path.exists(cfg.save_dir + 'output'):
            os.mkdir(cfg.save_dir + 'output')
        f = open(cfg.save_dir + 'output/result.txt', 'w')

        total_acc_cls = []
        total_tp_num = []
        total_all_num = []

        total_tps = []
        total_fps = []
        total_fns = []

        hist = np.zeros((cfg.NUM_OF_CLASSESS, cfg.NUM_OF_CLASSESS))

        import cv2
        for item in range(len(img_list)):
            k = int(cfg.seq_num / 2)
            valid_images = [cv2.imread(img_list[item])]
            label_key = imageio.imread(label_list[item])[:, int(cfg.IMAGE_ORI_WIDTH * k):int(cfg.IMAGE_ORI_WIDTH * (k+1))]
            valid_annotations = [np.expand_dims(label_key, axis=2)]
            im_name = img_list[item].split('/')[-1].split('.')[0]
            print(im_name)
            img_ori = valid_images[0]
            # --sequence
            img_in = img_ori[:, 0:cfg.IMAGE_ORI_WIDTH, :]
            for i in range(1, cfg.seq_num):
                img_in = np.concatenate([img_in, img_ori[:, int(i*cfg.IMAGE_ORI_WIDTH):int((i+1)*cfg.IMAGE_ORI_WIDTH), :]], axis=2)

            h_in, w_in, _ = img_in.shape

            scs = [0.45, 0.6, 0.75]
            # scs = [0.5]

            maps = []
            for sc in scs:
                img = cv2.resize(img_in, (int(float(w_in) * sc), int(float(h_in) * sc)), interpolation=cv2.INTER_LINEAR)
                score_map = data_crop_test_output(sess, image, logits, img, mean, std, mean_bgr, crop_size_h,
                                                  crop_size_w, stride)
                score_map = cv2.resize(score_map, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
                maps.append(score_map)
            score_map = np.mean(np.stack(maps), axis=0)

            maps2 = []
            for sc in scs:
                img2 = cv2.resize(img_in, (int(float(w_in) * sc), int(float(h_in) * sc)), interpolation=cv2.INTER_LINEAR)
                img2 = cv2.flip(img2, 1)
                score_map2 = data_crop_test_output(sess, image, logits, img2, mean, std, mean_bgr, crop_size_h,
                                                  crop_size_w, stride)
                score_map2 = cv2.resize(score_map2, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
                maps2.append(score_map2)
            score_map2 = np.mean(np.stack(maps2), axis=0)
            score_map2 = cv2.flip(score_map2, 1)
            score_map = (score_map + score_map2) / 2

            pred_label = np.argmax(score_map, 2)

            pred_label = np.asarray(pred_label, dtype='uint8')
            pred = [pred_label[:, :, np.newaxis]]

            hist += fast_hist(valid_annotations[0].flatten(), pred[0].flatten(), cfg.NUM_OF_CLASSESS)  # gt, pred, class

            pred_vision(pred[0], im_name, cfg.dataset)
            pred_vision(valid_annotations[0], im_name + '_gt', cfg.dataset)
            print('image ' + str(item))
            f.write('image ' + im_name + '\n')
            f.write('scales: ' + str(scs) + '\n')
            for itr in range(FLAGS.batch_size):

                cls_acc, img_acc, tp_num, all_num = eval_img2(valid_annotations[itr], pred[itr])
                tps, fps, fns = eval_fscore(valid_annotations[itr], pred[itr])

                for cls in range(len(cls_acc)):
                    print(cfg.class_names[cls] + ': ' + str(cls_acc[cls]))
                    f.write(cfg.class_names[cls] + ': ' + str(cls_acc[cls]) + '\n')
                print('img-' + im_name + ': ' + str(img_acc))
                f.write('img-' + im_name + ' : ' + str(img_acc))
                print('-----------------------------')
                f.write('-------------------------------' + '\n')
                print('\n')
                f.write('\n')

            total_acc_cls.append(cls_acc)
            total_tp_num.append(tp_num)
            total_all_num.append(all_num)

            total_tps.append(tps)
            total_fps.append(fps)
            total_fns.append(fns)

            # overall accuracy  1
            # print('Shape hist: ', hist.shape)

        f.write('Shape hist: ' + str(hist.shape) + '\n')
        over_acc = np.diag(hist).sum() / hist.sum()
        print('1 overall accuracy', over_acc)
        f.write('1 overall accuracy' + str(over_acc) + '\n')

        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print('1 mean accuracy', acc)
        f.write('1 mean accuracy' + str(acc) + '\n')

        # overall accuracy  2
        hist[0, :] = 0  # Ignore outlier
        # print('Shape hist: ', hist.shape)
        f.write('Shape hist: ' + str(hist.shape) + '\n')
        over_acc = np.diag(hist).sum() / hist.sum()
        print('2 overall accuracy', over_acc)
        f.write('2 overall accuracy' + str(over_acc) + '\n')

        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print('2 mean accuracy', acc)
        f.write('2 mean accuracy' + str(acc) + '\n')

        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f1-score
        f1_scores = []
        for c in range(1, cfg.NUM_OF_CLASSESS):
            TP = hist[c][c]
            FP = np.sum(hist[:, c]) - hist[c][c]
            FN = np.sum(hist[c, :]) - hist[c][c]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)

        mean_f1_score = sum(f1_scores) / len(f1_scores)
        print('f1_score: ' + str(mean_f1_score))
        f.write('f1 score: ' + str(mean_f1_score) + '\n')

        # per-class IU
        numerator = np.diag(hist)
        denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
        numerator_noBg = np.delete(numerator, 0, axis=0)
        denominator_noBg = np.delete(denominator, 0, axis=0)
        iu = numerator_noBg / denominator_noBg
        print('IoU ' + str(iu))
        f.write('IoU ' + str(iu) + '\n')
        print('mean IoU ', np.nanmean(iu))
        f.write('mean IoU ' + str(np.nanmean(iu)) + '\n')

        total_tps = np.array(total_tps)
        total_fps = np.array(total_fps)
        total_fns = np.array(total_fns)
        F1_socre2 = []
        for column in range(total_tps.shape[1]):

            cls_tp = []
            cls_fp = []
            cls_fn = []

            for row in range(total_tps.shape[0]):
                cls_tp.append(total_tps[row][column])
                cls_fp.append(total_fps[row][column])
                cls_fn.append(total_fns[row][column])
            prec = sum(cls_tp) / (sum(cls_tp) + sum(cls_fp))
            rec = sum(cls_tp) / (sum(cls_tp) + sum(cls_fn))
            # print(cfg.class_names[column] + '-prec:' + str(prec) + ', rec: ' + str(rec))
            F1_socre2.append((2 * prec * rec) / (prec + rec))
        # print('F1-score2: ' + str(sum(F1_socre2) / len(F1_socre2)))

        total_acc_cls = np.array(total_acc_cls)
        total_tp_num = np.array(total_tp_num)
        total_all_num = np.array(total_all_num)
        print('Total Accuracy: ')
        f.write('Total Accuracy: \n')

        filename = cfg.save_dir + 'output/acc.csv'
        f_csv = open(filename, 'w')
        writer = csv.writer(f_csv)

        class_avg_acc = []
        for column in range(total_acc_cls.shape[1]):

            cls_tp_num = []
            cls_all_num = []

            for row in range(total_acc_cls.shape[0]):
                cls_tp_num.append(total_tp_num[row][column])
                cls_all_num.append(total_all_num[row][column])

            class_acc = sum(cls_tp_num) / sum(cls_all_num)
            print(cfg.class_names[column] + '-acc:' + str(class_acc))
            f.write(cfg.class_names[column] + '-acc:' + str(class_acc) + '\n')
            writer.writerow([cfg.class_names[column], str(class_acc)])
            class_avg_acc.append(class_acc)

        print('\nTotal Acc:' + str(np.sum(total_tp_num) / np.sum(total_all_num)))
        f.write('\nTotal Acc:' + str(np.sum(total_tp_num) / np.sum(total_all_num)) + '\n')
        print('\nMean Acc:' + str(sum(class_avg_acc) / len(class_avg_acc)))
        f.write('\nMean Acc:' + str(sum(class_avg_acc) / len(class_avg_acc)) + '\n')

        writer.writerow(['Total acc', str(np.sum(total_tp_num) / np.sum(total_all_num))])
        writer.writerow(['Mean acc', str(sum(class_avg_acc) / len(class_avg_acc))])
        writer.writerow(['Mean_f1_score', str(mean_f1_score)])
        writer.writerow(['Mean IoU', str(np.nanmean(iu))])

        f_csv.close()
        f.close()



if __name__ == "__main__":
    tf.app.run()