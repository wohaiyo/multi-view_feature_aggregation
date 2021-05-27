# Config file for define some parameters
import argparse
import numpy as np
import tensorflow as tf
import math

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Facade ALK Network")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--image_height", type=int, default=544,
                        help="Image height and width of image.")
    parser.add_argument("--image_width", type=int, default=400,
                        help="Image height and width of image.")
    parser.add_argument("--learning_rate", type=float,
                        default=2e-4,
                        help="Learning rate for training.")
    parser.add_argument("--optimizer", type=str, default='Adam',    # Adam Momentum
                        help="optimizer for BP.")
    return parser.parse_args()
args = get_arguments()

# ---------------Modified Paras---------------------------
dataset = 'RueMonge428_seq3_homo'
use_gpu = '1'
seq_num = 3

NUM_OF_CLASSESS = 8

# The number of gradient accumulation
Gradient_Accumulation = 1

total_iter = 10000

model_save_num = 12

is_epoch_acc = False

is_time_acc = False
# 10s to show acc
acc_interval = 180

# epoch 10000 to show train data acc
start_show_iter = 2000

is_save_epoch = False
save_epoch_inter = 100
start_save_epoch = 500

is_save_step = True
save_step_inter = 2000
start_save_step = 8000

weight_decay = 0.0001

freeze_bn = True

is_save_last10_model = False
# ----------------------------------------------------------


if dataset == 'ecp_seq3':
    data_dir = 'data/ecp_seq3/'
    save_dir = 'saves/ecp_seq3/deeplabv3_plus/' # pspnet, deeplabv3_plus, danet
    logs_dir = 'tensorboard/ecp_seq3/'
    class_names = ['Outlier','Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 520

elif dataset == 'RueMonge428_seq3_homo':
    data_dir = 'data/RueMonge428_seq3_homo/'
    save_dir = 'saves/RueMonge428_seq3_homo_aggregation_att/'
    logs_dir = 'tensorboard/RueMonge428_seq3_homo/'
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Sky', 'Shop']
    train_number = 113


# ------------------------Other high-paras----------------------------------

pre_trained_model = 'data/pre-trained_model/resnet_v1_50.ckpt'

IMAGE_HEIGHT = args.image_height
IMAGE_WIDTH = args.image_width

IMAGE_ORI_HEIGHT = 1067     #
IMAGE_ORI_WIDTH = 800       # only used in image_reader

batch_size = args.batch_size

learning_rate = args.learning_rate

optimizer = args.optimizer

decay_rate = 0.9

summary_interval = 60  # 60s to save a summary

train_data_dir = data_dir + 'train'
train_data_list = data_dir + 'train.txt'

test_data_dir = data_dir + 'val'
test_data_list = data_dir + 'val.txt'

val_data_dir = data_dir + 'train'
val_data_list = data_dir + 'train.txt'

random_resize = False
random_color = False

if 'ecp' in train_data_list:
    random_scale = False
    random_mirror = False
else:
    random_scale = True
    random_mirror = True  # Intrinsics paras not suit flipping

minScale = 0.4                     # 0.75: ecp, 0.4: rue                            Modified
maxScale = 0.8                     # 1.25: ecp, 0.8: rue
random_crop_pad = True
ignore_label = 0

IMG_MEAN = np.array([103.94, 116.78, 123.68], dtype=np.float32)     # B G R


# -----------------Learning Schedule------------------------------

def get_cur_lr(step_ph):
    cur_lr = tf.py_func(_get_cur_lr, [step_ph], tf.float32)
    return cur_lr
def _get_cur_lr(step_ph):
    step = np.array(step_ph, np.int32)
    ep = int(step / (train_number / batch_size))
    if ep < 10:
        cur_lr = 1e-4
    elif ep < 20:
        cur_lr = 1e-5
    else:
        cur_lr = 1e-6

    return np.asarray(cur_lr, dtype=np.float32)

def get_step_lr(step_ph):
    step_lr = tf.py_func(_get_step_lr, [step_ph], tf.float32)
    return step_lr
def _get_step_lr(step_ph):
    step = np.array(step_ph, np.int32)
    ep = step
    if ep < 2000:
        step_lr = 2e-4
    elif ep < 4000:
        step_lr = 1e-4
    elif ep < 6000:
        step_lr = 5e-5
    elif ep < 8000:
        step_lr = 1e-5
    elif ep < 10000:
        step_lr = 1e-6
    else:
        step_lr = 1e-6

    return np.asarray(step_lr, dtype=np.float32)

def get_cosine_lr(step_ph):
    cur_lr = tf.py_func(_get_cosine_lr, [step_ph], tf.float32)
    return cur_lr

def _get_cosine_lr(step_ph):
    step = np.array(step_ph, np.int32)
    total_step = int((train_number / batch_size) * args.epoch_num)
    cur_lr = ((1 + math.cos((step * 3.1415926535897932384626433) / total_step)) * args.learning_rate) / 2
    return np.asarray(cur_lr, dtype=np.float32)

def noam_scheme(cur_step):                                  # warmup learning rate
    lr = tf.py_func(_noam_scheme, [cur_step], tf.float32)
    return lr

def _noam_scheme(cur_step):
    """
    if cur < warnup_step, lr increase
    if cur > warnup_step, lr decrease
    """
    step = np.array(cur_step, np.int32)
    init_lr = learning_rate
    global_step = total_iter
    warnup_factor = 1.0 / 3
    power = 0.9
    warnup_step = 500

    if step <= warnup_step:
        alpha = step / warnup_step
        warnup_factor = warnup_factor * (1 - alpha) + alpha
        lr = init_lr * warnup_factor

    else:
        # learning_rate = tf.scalar_mul(init_lr, tf.pow((1 - cur_step / global_step), power))
        lr = init_lr * np.power(
            (1 - (step - warnup_step) / (global_step - warnup_step)), power)

    return np.asarray(lr, dtype=np.float32)

def circle_scheme(cur_step):                                # circle learning rate
    lr = tf.py_func(_circle_scheme, [cur_step], tf.float32)
    return lr

def _circle_scheme(cur_step):
    step = np.array(cur_step, np.int32)
    CYCLE = 1000
    LR_INIT = learning_rate
    LR_MIN = 1e-10
    scheduler = lambda x: ((LR_INIT - LR_MIN) / 2) * (np.cos(3.1415926535897932384626433 * (np.mod(x - 1, CYCLE) / (CYCLE))) + 1) + LR_MIN
    lr = scheduler(step)

    return np.asarray(lr, dtype=np.float32)
