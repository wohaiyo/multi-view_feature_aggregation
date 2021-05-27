import cv2
import imageio
import numpy as np
import os
import csv

img_height = 1067
img_width = 800
seq_num = 5
dataset_name = 'RueMonge428'

train_img_path = dataset_name + '/train/img/'
train_lab_path = dataset_name + '/train/label/'
train_dep_path = dataset_name + '/train/depth/'
train_pos_path = dataset_name + '/train/pose/'
train_ins_path = dataset_name + '/train/intrinsics/'

val_img_path = dataset_name + '/val/img/'
val_lab_path = dataset_name + '/val/label/'
val_dep_path = dataset_name + '/val/depth/'
val_pos_path = dataset_name + '/val/pose/'
val_ins_path = dataset_name + '/val/intrinsics/'


save_train_dir = dataset_name + '_seq' + str(seq_num) + '_homo/train/'
save_val_dir = dataset_name + '_seq' + str(seq_num) + '_homo/val/'

if not os.path.exists(save_train_dir + '/img'):
    os.makedirs(save_train_dir + '/img')
if not os.path.exists(save_train_dir + '/label'):
    os.makedirs(save_train_dir + '/label')
if not os.path.exists(save_train_dir + '/label_color'):
    os.makedirs(save_train_dir + '/label_color')
if not os.path.exists(save_train_dir + '/depth'):
    os.makedirs(save_train_dir + '/depth')
if not os.path.exists(save_train_dir + '/pose'):
    os.makedirs(save_train_dir + '/pose')
if not os.path.exists(save_train_dir + '/intrinsics'):
    os.makedirs(save_train_dir + '/intrinsics')

if not os.path.exists(save_val_dir + '/img'):
    os.makedirs(save_val_dir + '/img')
if not os.path.exists(save_val_dir + '/label'):
    os.makedirs(save_val_dir + '/label')
if not os.path.exists(save_val_dir + '/label_color'):
    os.makedirs(save_val_dir + '/label_color')
if not os.path.exists(save_val_dir + '/depth'):
    os.makedirs(save_val_dir + '/depth')
if not os.path.exists(save_val_dir + '/pose'):
    os.makedirs(save_val_dir + '/pose')
if not os.path.exists(save_val_dir + '/intrinsics'):
    os.makedirs(save_val_dir + '/intrinsics')

# sift
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des

def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut, H, status

def siftImageAlignment_label(H, label):
    '''
    Homography matrix for label using nearst inter
    :param label1:
    :return:
    '''

    imgOut = cv2.warpPerspective(label, H, (label.shape[1], label.shape[0]),
                                 flags=cv2.INTER_NEAREST)# + cv2.WARP_INVERSE_MAP)
    return imgOut

def sorted_aphanumeric(data):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def readCSV(filename):
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            return row

def writeCSV(filename, lists):
    write_lists = []
    for list in lists:
        for l in list:
            write_lists.append(l)

    f_csv = open(filename, 'w')
    writer = csv.writer(f_csv)
    writer.writerow(write_lists)


def save_imgs(path, numbers):
    name = str(numbers[int(seq_num / 2)])
    save_img = np.zeros((img_height, int(img_width * seq_num), 3), dtype=np.uint8)
    save_label = np.zeros((img_height, int(img_width * seq_num)), dtype=np.uint8)
    save_depth = np.zeros((img_height, int(img_width * seq_num)), dtype=np.uint16)
    poses = []
    intrinsicses = []

    if 'tra' in path:
        tgt_img = cv2.imread(train_img_path + 'IMG_' + name + '.jpg')
        tgt_label = imageio.imread(train_lab_path + 'IMG_' + name + '.png')
        tgt_depth = cv2.imread(train_dep_path + 'IMG_' + name + '.png', cv2.COLOR_BGR2GRAY) / 1000
        pose = readCSV(train_pos_path + 'IMG_' + name + '.csv')
        ins = readCSV(train_ins_path + 'IMG_' + name + '.csv')
        for i, num in enumerate(numbers):
            if num < 0:
                img = tgt_img
                label = tgt_label
                depth = tgt_depth
                poses.append(pose)
                intrinsicses.append(ins)
            elif i == int(seq_num / 2):
                img = tgt_img
                label = tgt_label
                depth = tgt_depth
                poses.append(pose)
                intrinsicses.append(ins)
            else:
                img = cv2.imread(train_img_path + 'IMG_' + str(num) + '.jpg')
                img, H, _ = siftImageAlignment(tgt_img, img)
                label = tgt_label # siftImageAlignment_label(H, tgt_label)

                depth = cv2.imread(train_dep_path + 'IMG_' + str(num) + '.png', cv2.COLOR_BGR2GRAY)
                src_pose = readCSV(train_pos_path + 'IMG_' + str(num) + '.csv')
                poses.append(src_pose)
                src_intrinsics = readCSV(train_ins_path + 'IMG_' + str(num) + '.csv')
                intrinsicses.append(src_intrinsics)


            # resize train imgs to [h, w]
            img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            save_img[0:img_height, int(i * img_width):int((i + 1) * img_width), :] = img

            label = cv2.resize(label, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            save_label[0:img_height, int(i * img_width):int((i + 1) * img_width)] = label

            depth = cv2.resize(depth, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            save_depth[0:img_height, int(i * img_width):int((i + 1) * img_width)] = depth

        cv2.imwrite(save_train_dir + 'img/IMG_' + name + '.jpg', save_img)
        imageio.imsave(save_train_dir + 'label/IMG_' + name + '.png', save_label)
        imageio.imsave(save_train_dir + 'label_color/IMG_' + name + '.png', save_label*20)
        cv2.imwrite(save_train_dir + 'depth/IMG_' + name + '.png', save_depth)
        writeCSV(save_train_dir + 'pose/IMG_' + name + '.csv', poses)
        writeCSV(save_train_dir + 'intrinsics/IMG_' + name + '.csv', intrinsicses)

    elif 'val' in path:
        tgt_img = cv2.imread(val_img_path + 'IMG_' + name + '.jpg')
        tgt_label = imageio.imread(val_lab_path + 'IMG_' + name + '.png')
        tgt_depth = cv2.imread(val_dep_path + 'IMG_' + name + '.png', cv2.COLOR_BGR2GRAY) / 1000
        pose = readCSV(val_pos_path + 'IMG_' + name + '.csv')
        ins = readCSV(val_ins_path + 'IMG_' + name + '.csv')
        for i, num in enumerate(numbers):
            if num < 0:
                img = tgt_img
                label = tgt_label
                depth = tgt_depth
                poses.append(pose)
                intrinsicses.append(ins)
            elif i == int(seq_num / 2):
                img = tgt_img
                label = tgt_label
                depth = tgt_depth
                poses.append(pose)
                intrinsicses.append(ins)
            else:
                img = cv2.imread(val_img_path + 'IMG_' + str(num) + '.jpg')
                img, H, _ = siftImageAlignment(tgt_img, img)
                label = tgt_label # siftImageAlignment_label(H, tgt_label)

                depth = cv2.imread(val_dep_path + 'IMG_' + str(num) + '.png', cv2.COLOR_BGR2GRAY)
                src_pose = readCSV(val_pos_path + 'IMG_' + str(num) + '.csv')
                poses.append(src_pose)
                src_intrinsics = readCSV(val_ins_path + 'IMG_' + str(num) + '.csv')
                intrinsicses.append(src_intrinsics)


            # resize val imgs to [h, w]
            img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            save_img[0:img_height, int(i * img_width):int((i + 1)*img_width), :] = img

            label = cv2.resize(label, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            save_label[0:img_height, int(i * img_width):int((i + 1) * img_width)] = label

            depth = cv2.resize(depth, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            save_depth[0:img_height, int(i * img_width):int((i + 1) * img_width)] = depth


        cv2.imwrite(save_val_dir + 'img/IMG_' + name + '.jpg', save_img)
        imageio.imsave(save_val_dir + 'label/IMG_' + name + '.png', save_label)
        imageio.imsave(save_val_dir + 'label_color/IMG_' + name + '.png', save_label*20)
        cv2.imwrite(save_val_dir + 'depth/IMG_' + name + '.png', save_depth)
        writeCSV(save_val_dir + 'pose/IMG_' + name + '.csv', poses)
        writeCSV(save_val_dir + 'intrinsics/IMG_' + name + '.csv', intrinsicses)

# train files
train_files = os.listdir(train_img_path)
train_files = sorted_aphanumeric(train_files)
train_nums = []
for tra in train_files:
    num = int(tra.split('_')[-1].split('.')[0])
    train_nums.append(num)

# val files
val_files = os.listdir(val_img_path)
val_files = sorted_aphanumeric(val_files)
val_nums = []
for val in val_files:
    num = int(val.split('_')[-1].split('.')[0])
    val_nums.append(num)

train_count = 0
# train
for t in train_nums:
    cur_num = t
    view_nums = []
    # Before
    for n in range(int(seq_num / 2), 0, -1):
        before_num = cur_num - n
        if before_num in train_nums:
            view_nums.append(before_num)
        else:
            view_nums.append(-1)
            train_count += 1
    # Current
    view_nums.append(cur_num)
    # After
    for n in range(1, seq_num - int(seq_num / 2)):
        after_num = cur_num + n
        if after_num in train_nums:
            view_nums.append(after_num)
        else:
            view_nums.append(-1)
            train_count += 1

    save_imgs(save_train_dir, view_nums)
    print(t)
print('Train images finished!')
print('train lack count: ' + str(train_count))

val_count = 0
# val
for v in val_nums:
    cur_num = v
    view_nums = []
    # Before
    for n in range(int(seq_num / 2), 0, -1):
        before_num = cur_num - n
        if before_num in val_nums:
            view_nums.append(before_num)
        else:
            view_nums.append(-1)
            val_count += 1
    # Current
    view_nums.append(cur_num)
    # After
    for n in range(1, seq_num - int(seq_num / 2)):
        after_num = cur_num + n
        if after_num in val_nums:
            view_nums.append(after_num)
        else:
            view_nums.append(-1)
            val_count += 1

    save_imgs(save_val_dir, view_nums)
    print(v)
print('Val images finished!')
print('Val lack count: ' + str(val_count))
print('Finished!')

# generate txt files
data_dir = dataset_name + '_seq' + str(seq_num) + '_homo/'
tra_path = data_dir + 'train/img/*.jpg'
import glob
im_list = glob.glob(tra_path)
f = open(data_dir + 'train.txt', 'w')
for im_name in im_list:
    name = im_name.split('/')[-1].split('.')[0]
    im_path = '/img/'+name+'.jpg'
    label_path = '/label/'+name+'.png'
    depth_path = '/depth/'+name+'.png'
    pose_path = '/pose/' + name + '.csv'
    ins_path = '/intrinsics/' + name + '.csv'

    f.write(im_path + ' ' + label_path + ' ' + depth_path + ' ' + pose_path + ' ' + ins_path +'\n')

f.close()
print('Finished image generation!')

val_path = data_dir + 'val/img/*.jpg'
im_list = glob.glob(val_path)
f = open(data_dir + 'val.txt', 'w')
for im_name in im_list:
    name = im_name.split('/')[-1].split('.')[0]
    im_path = '/img/'+name+'.jpg'
    label_path = '/label/'+name+'.png'
    depth_path = '/depth/'+name+'.png'
    pose_path = '/pose/' + name + '.csv'
    ins_path = '/intrinsics/' + name + '.csv'

    f.write(im_path + ' ' + label_path + ' ' + depth_path + ' ' + pose_path + ' ' + ins_path +'\n')

f.close()
print('Finished image_list!')

