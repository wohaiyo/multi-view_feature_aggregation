import cv2
import config as cfg
import numpy as np
import os

def eval_img2(gt, pred):
    cls_acc = []
    TP_num = []
    pos_all_total = []


    for i in range(1, cfg.NUM_OF_CLASSESS):   # ignore 0
        gt_i = np.zeros(gt.shape, np.int)
        pred_i = np.zeros(gt.shape, np.int)
        zero = np.zeros(gt.shape, np.int)

        gt_i[gt == i] = 1
        pred_i[pred == i] = 1

        cls_i = gt_i.copy()
        cls_i[gt != i] = -1
        zero[cls_i == pred_i] = 1

        TP = np.sum(zero)

        pos_all = np.sum(gt_i)

        if pos_all == 0:
            cls_acc.append(-1)
        else:
            cls_acc.append(TP / pos_all)
        TP_num.append(TP)
        pos_all_total.append(pos_all)

    return cls_acc, sum(TP_num) / sum(pos_all_total),  TP_num, pos_all_total

def eval_fscore(gt, pred):
    TPs = []
    FPs = []
    FNs = []

    for i in range(1, cfg.NUM_OF_CLASSESS):   # ignore 0
        gt_i = np.zeros(gt.shape, np.int)
        pred_i = np.zeros(gt.shape, np.int)
        zero_tp = np.zeros(gt.shape, np.int)

        # TP
        gt_i[gt == i] = 1
        pred_i[pred == i] = 1

        cls_i = gt_i.copy()
        cls_i[gt != i] = -1
        zero_tp[cls_i == pred_i] = 1

        TP = np.sum(zero_tp)

        # FP
        gt_i[gt != i] = 1
        gt_i[gt == i] = 0
        gt_i[gt == 0] = 0

        pred_i[pred == i] = 1
        pred_i[pred != i] = 0

        FP = np.sum(gt_i * pred_i)

        # FN
        gt_i[gt != i] = 0
        gt_i[gt == i] = 1
        gt_i[gt == 0] = 0

        pred_i[pred == i] = 0
        pred_i[pred != i] = 1

        FN = np.sum(gt_i * pred_i)

        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)

    return TPs, FPs, FNs

def pred_vision(pred, name, dataset):  # pred （h, w， 1)
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_new = np.zeros([height, width, 3], dtype=np.uint8)
    pred_B = np.zeros([height, width, 1], dtype=np.uint8)
    pred_G = np.zeros([height, width, 1], dtype=np.uint8)
    pred_R = np.zeros([height, width, 1], dtype=np.uint8)

    if 'etrims' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 128],
                       [2, 128, 0, 128],
                       [3, 0, 128, 128],
                       [4, 128, 128, 128],
                       [5, 0, 64, 128],
                       [6, 128, 128, 0],
                       [7, 0, 128, 0],
                       [8, 128, 0, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    elif 'cmp' in dataset:      # ['Background','Door', 'Shop', 'Balcony', 'Window', 'Wall']    # (0,0,0) ()
        label_color = [[0, 0, 0, 0],
                       [1, 255, 170, 0],
                       [2, 0, 0, 170],
                       [3, 85, 255, 170],
                       [4, 255, 85, 0],
                       [5, 255, 0, 0]
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    elif 'art' in dataset or 'Art' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 128, 255],
                       [2, 0, 255, 0],
                       [3, 255, 0, 128],
                       [4, 0, 0, 255],
                       [5, 0, 255, 255],
                       [6, 255, 255, 128],
                       [7, 255, 0, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]
    elif 'camvid' in dataset:
        label_color = [[0, 0, 0, 0],
                                [1, 128, 128, 128],  # R G B
                                [2, 128, 0, 0],
                                [3, 192, 192, 192],
                                [4, 128, 64, 128],
                                [5, 60, 40, 222],
                                [6, 128, 128, 0],
                                [7, 192, 128, 128],
                                [8, 64, 64, 128],
                                [9, 64, 0, 128],
                                [10, 64, 64, 0],
                                [11, 0, 128, 192]]

        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][3]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][1]

    elif 'Rue' in dataset or 'Monge' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 255],
                       [2, 0, 255, 255],
                       [3, 255, 0, 128],
                       [4, 0, 128, 255],
                       [5, 255, 0, 0],
                       [6, 255, 255, 128],
                       [7, 0, 255, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    else:                                               # ECP
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 255],
                       [2, 0, 255, 255],
                       [3, 255, 0, 128],
                       [4, 0, 128, 255],
                       [5, 255, 0, 0],
                       [6, 128, 128, 128],
                       [7, 255, 255, 128],
                       [8, 0, 255, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    pred_new = np.concatenate([pred_B, pred_G, pred_R], 2)
    save_name = cfg.save_dir + 'output/' + name + '.png'
    cv2.imwrite(save_name, pred_new)
    print(save_name + ' is saved.')




