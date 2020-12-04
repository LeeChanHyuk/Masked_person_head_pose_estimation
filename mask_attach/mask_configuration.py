import cv2
import os
import numpy as np

DATA_DIR = '/home/leechanhyuk/Downloads/NEW_IMAGE'
MASK_DIR = '/home/leechanhyuk/Downloads/mask1'
f = open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt', 'r')
g = open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_pose_list.txt', 'r')
h = open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_coordinate_list.txt', 'r')
count = 0


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]
    return img_crop


"""while(True):
    name = f.readline()
    name = name[0:len(name)-1]
    print(name)
    if name is not None:
        count = count+1
        # image connect
        image = cv2.imread(DATA_DIR+'/'+name+'.jpg')
        image2 = image.copy()
        (h,w)=image.shape[:2]
        pose = g.readline()
        (yaw,pitch,roll) = pose.split()
        yaw = int(float(yaw)/10)
        pitch = int(float(pitch)/10)
        roll = int(float(roll)/10)
        yaw = yaw*10
        pitch = pitch*10
        roll=roll*10
"""
k = open('/home/leechanhyuk/Downloads/mask1/list.txt', 'w')
for name in os.listdir(MASK_DIR+'/mask2'):
    if name is not None:
        try:
            # mask detection
            mask = cv2.imread(MASK_DIR + '/mask2' + '/' + name)
            mask_gray = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
            # mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
            mask2 = mask.copy()
            ret,mask_area = cv2.threshold(mask_gray,1,255,cv2.THRESH_BINARY) # mask is the only region which is white
            coutours = cv2.findContours(mask_area,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            maxs=0
            max_index=0
            for i in range(len(coutours[0])):
                if len(coutours[0][i])>maxs:
                    a= coutours[0][i]
                    maxs = len(coutours[0][i])
                    max_index=i
            max_contours = coutours[0][max_index]
            rect = cv2.minAreaRect(max_contours)

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            min_x = min(box[0][0],box[1][0],box[2][0],box[3][0])
            max_x = max(box[0][0],box[1][0],box[2][0],box[3][0])
            min_y = min(box[0][1], box[1][1], box[2][1], box[3][1])
            max_y = max(box[0][1], box[1][1], box[2][1], box[3][1])
            cv2.drawContours(mask,[box],0,(255,255,200),1)
            mask_crop = mask2[min_y:max_y+1 , min_x:max_x+1]
            """black_mask_crop = mask_crop.copy()
            (crop_h,crop_w) = mask_crop.shape[:2]
            ret,mask_croped_thresholded = cv2.threshold(mask_crop.copy(),1,255,cv2.THRESH_BINARY)
            max_y = 0
            white_pixel=[]
            white_pixel_count=0
            for i in range(crop_w):
                for j in range(crop_h):
                    if mask_croped_thresholded[j,i] == 255:
                        white_pixel_count=white_pixel_count+1
                white_pixel.append(white_pixel_count)
                white_pixel_count=0
            white_pixel_max = max(white_pixel)
            for i in range(crop_w):
                for j in range(crop_h):
                    if mask_croped_thresholded[j,i] == 255:
                        white_pixel_count=white_pixel_count+1
                if white_pixel_count<(white_pixel_max/2):
                    for j in range(crop_h):
                        mask_croped_thresholded[j, i] = 0
                white_pixel_count=0

            kernel = np.ones((5, 5), np.uint8)
            mask_croped_thresholded = cv2.morphologyEx(mask_croped_thresholded, cv2.MORPH_OPEN, kernel)
            cv2.imshow("a", mask_croped_thresholded)
            cv2.bitwise_and(mask_crop,mask_croped_thresholded,mask_crop)

            f_coutours = cv2.findContours(mask_croped_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            f_maxs = 0
            f_max_index = 0
            for i in range(len(f_coutours[0])):
                if len(f_coutours[0][i]) > f_maxs:
                    a = f_coutours[0][i]
                    f_maxs = len(f_coutours[0][i])
                    f_max_index = i
            f_max_contours = f_coutours[0][f_max_index]
            f_rect = cv2.minAreaRect(f_max_contours)

            f_box = cv2.boxPoints(f_rect)
            f_box = np.int0(f_box)
            f_min_x = min(f_box[0][0], f_box[1][0], f_box[2][0], f_box[3][0])
            f_max_x = max(f_box[0][0], f_box[1][0], f_box[2][0], f_box[3][0])
            f_min_y = min(f_box[0][1], f_box[1][1], f_box[2][1], f_box[3][1])
            f_max_y = max(f_box[0][1], f_box[1][1], f_box[2][1], f_box[3][1])
            f_mask_crop = mask_crop[f_min_y:f_max_y + 1, f_min_x:f_max_x + 1]

            cv2.imshow("f_mask_crop",f_mask_crop)"""
            cv2.imwrite('/home/leechanhyuk/Downloads/mask1/temp/' + name, mask_crop)
            # inverse_mask = cv2.bitwise_not(mask_area) # mask it the only region which is black
            # cv2.imshow("image2", mask)
            # cv2.bitwise_and(image2,inverse_mask,image) # mask region in the image is black
            # cv2.add(image2,mask,image2)
        except:
            print(name)
            mask = cv2.imread(MASK_DIR + '/mask2' + '/' + name)
            cv2.imwrite('/home/leechanhyuk/Downloads/mask1/temp2/' + name, mask)




        # mouth detection
        # if mouth_detection is not None:
