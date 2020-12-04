import cv2
import os
import numpy as np

DATA_DIR = '/home/leechanhyuk/Downloads/NEW_IMAGE'
MASK_DIR = '/home/leechanhyuk/Downloads/mask1'
MASK_SAVE_DIR = '/home/leechanhyuk/Downloads/term_project/deep-head-pose-master/code/output/snapshots/mask_img'
f = open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list_4805.txt', 'a')
g = open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_pose_list_4805.txt', 'a')
h = open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_coordinate_list_4805.txt', 'a')



def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

filename_lists = get_list_from_filenames('/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt')
coordinate_lists = get_list_from_filenames('/home/leechanhyuk/PycharmProjects/tensorflow1/file_coordinate_list.txt')
pose_lists = get_list_from_filenames('/home/leechanhyuk/PycharmProjects/tensorflow1/file_pose_list.txt')
count=0
for i in range(5000,15000):
        img = cv2.imread(DATA_DIR+'/'+filename_lists[i]+'.jpg')
        # Crop the face loosely
        pt2d = coordinate_lists[i].split(' ')
        x_min = int(pt2d[0])
        y_min = int(pt2d[2])
        x_max = int(pt2d[1])
        y_max = int(pt2d[3])



        # k = 0.2 to 0.40
        # We get the pose in radians
        pre_pose = pose_lists[i].split(' ')
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

        #cv2.imwrite(MASK_SAVE_DIR+'/'+filename_lists[i]+'.jpg',img)
        width=5
        height=11
        x_offset=0
        y_offset=0
        x_plus=0
        y_plus=0
        while(1):
                try:
                        img2 = img.copy()
                        crop_height_half = int((y_max - y_min + 1) * width / height)+y_plus
                        crop_width = x_max-x_min-30+1+x_plus
                        resized_mask = cv2.resize(mask, dsize=(int(crop_width)+1, int(crop_height_half)+1))
                        ret, mask_area = cv2.threshold(resized_mask, 100, 255,
                                                       cv2.THRESH_BINARY)  # mask is the only region which is white
                        cv2.bitwise_not(mask_area, mask_area)
                        img_mask = img2[y_max-crop_height_half+y_offset:y_max+1+y_offset, x_max-crop_width+x_offset:x_max+1+x_offset]
                        img_mask = cv2.bitwise_and(img_mask, mask_area)
                        img_mask = cv2.bitwise_or(img_mask, resized_mask)
                        img2[y_max-crop_height_half+y_offset:y_max+1+y_offset, x_max-crop_width+x_offset:x_max+1+x_offset] = img_mask
                        crop_img = img2[y_min:y_max+1,x_min:x_max+1]
                        cv2.imshow("img",crop_img)
                        input_key = cv2.waitKey(0) & 0xFF
                        print(input_key)
                        if input_key==ord('7'):
                                x_offset=x_offset-10
                                y_offset=y_offset-10
                        elif input_key==ord('8'):
                                y_offset=y_offset-10
                        elif input_key==ord('9'):
                                x_offset=x_offset+10
                                y_offset=y_offset-10
                        elif input_key==ord('4'):
                                x_offset=x_offset-10
                        elif input_key==ord('6'):
                                x_offset=x_offset+10
                        elif input_key==ord('1'):
                                x_offset=x_offset-10
                                y_offset=y_offset+10
                        elif input_key==ord('2'):
                                y_offset=y_offset+10
                        elif input_key==ord('3'):
                                x_offset=x_offset+10
                        elif input_key== ord('5'):
                                f.write(filename_lists[i]+'\n')
                                g.write(pose_lists[i]+'\n')
                                h.write(coordinate_lists[i]+'\n')
                                cv2.imwrite('/home/leechanhyuk/github/mask_dataset/temp/' + filename_lists[i] + '.jpg',crop_img)  # synthesize mask
                                break
                        elif input_key == ord('t'):
                                f.close()
                                g.close()
                                h.close()
                                break
                        elif input_key == ord('+'):
                                x_plus=x_plus+10
                                y_plus=y_plus+10
                        elif input_key == ord('-'):
                                x_plus=x_plus-10
                                y_plus=y_plus-10
                        else:
                                break
                except:
                        break


# 523장까지 처리