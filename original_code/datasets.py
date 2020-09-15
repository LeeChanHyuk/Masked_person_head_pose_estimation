import os
import numpy as np
import cv2
import pandas as pd
import torch
from PIL import Image, ImageFilter
import utils
import random
def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines
class Synhead():
    def __init__(self, data_dir, csv_path, transform, test=False):
        column_names = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.X_train = tmp_df['path']
        self.y_train = tmp_df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']]
        self.length = len(tmp_df)
        self.test = test
    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.X_train.iloc[index]).strip('.jpg') + '.png'
        img = Image.open(path)
        img = img.convert('RGB')
        x_min, y_min, x_max, y_max, yaw, pitch, roll = self.y_train.iloc[index]
        x_min = float(x_min); x_max = float(x_max)
        y_min = float(y_min); y_max = float(y_max)
        yaw = -float(yaw); pitch = float(pitch); roll = float(roll)
        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        width, height = img.size
        # Crop the face
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        return self.length
class Pose_300W_LP():
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
        self.filename_lists = get_list_from_filenames('/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt')
        self.coordinate_lists = get_list_from_filenames('/home/leechanhyuk/PycharmProjects/tensorflow1/file_coordinate_list.txt')
        self.pose_lists = get_list_from_filenames('/home/leechanhyuk/PycharmProjects/tensorflow1/file_pose_list.txt')

    def __getitem__(self, index):
        """img = cv2.imread(self.data_dir+'/'+self.filename_lists[index]+'.jpg')
        # Crop the face loosely
        pt2d = self.coordinate_lists[index].split(' ')
        x_min = int(pt2d[0])
        y_min = int(pt2d[2])
        x_max = int(pt2d[1])
        y_max = int(pt2d[3])



        # k = 0.2 to 0.40
        # We get the pose in radians
        pre_pose = self.pose_lists[index].split(' ')
        # And convert to degrees.
        pre_yaw = int(float(pre_pose[0])/10)
        pre_pitch = int(float(pre_pose[1])/10)
        pre_roll = int(float(pre_pose[2])/10)
        pre_yaw = pre_yaw * 10
        pre_pitch = pre_pitch * 10
        pre_roll = pre_roll * 10
        pose_name = str(int(pre_yaw)) + ' ' + str(int(pre_pitch)) + ' ' + str(int(pre_roll))
        mask = cv2.imread('/home/leechanhyuk/Downloads/mask1/mask2_converted/' + pose_name + '.jpg')

        img = np.array(img)
        img = img[y_min:y_max + 1, x_min:x_max + 1]
        (i_h, i_w) = img.shape[:2]
        crop_height_half = int(((y_max - y_min + 1)*3)/5)
        crop_width = i_w
        mask = cv2.resize(mask, dsize=(int(crop_width), int(crop_height_half)))
        ret, mask_area = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)  # mask is the only region which is white
        cv2.bitwise_not(mask_area, mask_area)
        img_mask = img[(i_h - crop_height_half):i_h, 0:crop_width]
        img_mask = cv2.bitwise_and(img_mask, mask_area)
        img_mask = cv2.bitwise_or(img_mask, mask)
        img[(i_h - crop_height_half):i_h, 0:crop_width] = img_mask
        img = Image.fromarray(img)
        """

        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))

        # Crop the face loosely
        pt2d = self.coordinate_lists[index].split(' ')
        x_min = int(pt2d[0])
        y_min = int(pt2d[2])
        x_max = int(pt2d[1])
        y_max = int(pt2d[3])
        img = img[y_min:y_max + 1, x_min:x_max + 1]

        """random_value = random.randrange(0,3)
        road = random_value%3
        if road ==0:
            img = img.crop((0,0,x_size,int(y_size*3/5)))
        elif road == 1:
            img = img.crop((0,int(y_size*1/5),x_size,int(y_size*4/5)))
        elif road == 2:
            img = img.crop((0,int(y_size*2/5),x_size,int(y_size)))
        px = img.load()
        if x_ran+30>x_size:
            x_box=x_size-x_ran
        else:
            x_box=30
        if y_ran+30>y_size:
            y_box = y_size-y_ran
        else:
            y_box=30

        for i in range(x_ran, x_ran+x_box):
            for j in range(y_ran, y_ran+y_box):
                px[i, j] = (255, 255, 255)"""

        # We get the pose in radians
        pose = self.pose_lists[index].split(' ')
        # And convert to degrees.
        yaw = int(float(pose[0]))
        pitch = int(float(pose[1]))
        roll = int(float(pose[2]))
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = cv2.flip(img,1)
        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.resize(img,dsize=(128,128))
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        img = Image.fromarray(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # 122,450
        return self.length
class Pose_300W_LP_random_ds():
    # 300W-LP dataset with random downsampling
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        ds = 1 + np.random.randint(0,4) * 5
        original_size = img.size
        img = img.resize((img.size[0] / ds, img.size[1] / ds), resample=Image.NEAREST)
        img = img.resize((original_size[0], original_size[1]), resample=Image.NEAREST)
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # 122,450
        return self.length
class AFLW2000():
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        px = img.load()

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # 2,000
        return self.length
class AFLW2000_ds():
    # AFLW2000 dataset with fixed downsampling
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        ds = 3  # downsampling factor
        original_size = img.size
        img = img.resize((img.size[0] / ds, img.size[1] / ds), resample=Image.NEAREST)
        img = img.resize((original_size[0], original_size[1]), resample=Image.NEAREST)
        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # 2,000
        return self.length
class AFLW_aug():
    # AFLW dataset with flipping
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Augment
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length
class AFLW():
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length
class AFW():
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]
        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]
        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)
        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # Around 200
        return self.length
class BIWI():
    def __init__(self, data_dir, filename_path, transform, img_ext='.png', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext))
        img = img.convert(self.image_mode)
        pose_path = os.path.join(self.data_dir, self.y_train[index] + '_pose' + self.annot_ext)
        y_train_list = self.y_train[index].split('/')
        bbox_path = os.path.join(self.data_dir, y_train_list[0] + '/dockerface-' + y_train_list[-1] + '_rgb' + self.annot_ext)
        # Load bounding box
        bbox = open(bbox_path, 'r')
        line = bbox.readline().split(' ')
        if len(line) < 4:
            x_min, y_min, x_max, y_max = 0, 0, img.size[0], img.size[1]
        else:
            x_min, y_min, x_max, y_max = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        bbox.close()
        # Load pose in degrees
        pose_annot = open(pose_path, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)
        R = np.array(R)
        T = R[3,:]
        R = R[:3,:]
        pose_annot.close()
        R = np.transpose(R)
        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
        # Loosely crop face
        k = 0.35
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, self.X_train[index]
    def __len__(self):
        # 15,667
        return self.length
