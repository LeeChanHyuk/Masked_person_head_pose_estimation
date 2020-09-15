import sys, os, argparse
sys.path.append("/home/leechanhyuk/Downloads/term_project/deep-head-pose-master")
sys.path.append("/home/leechanhyuk/Downloads/term_project/deep-head-pose-master/code")

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from multiprocessing import Process, Queue
from mark_detector import MarkDetector
from mark_detector import FaceDetector



import datasets, hopenet, utils

from skimage import io

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cuda0 = torch.device('cuda:0')
    cudnn.enabled = True
    mark_detector = MarkDetector()
    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path
    transformations = transforms.Compose([transforms.Scale(240),
                                          transforms.RandomCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pose_dataset = datasets.Pose_300W_LP('/home/leechanhyuk/Downloads/NEW_IMAGE',
                                         '/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list_4805.txt',
                                         transformations)
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=1,  # 원래는 arg_parse로 받는 수 였음(수정함)
                                               shuffle=True,
                                               num_workers=2)

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 67)

    # Dlib face detection model

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load("/home/leechanhyuk/Downloads/term_project/deep-head-pose-master/code/output/snapshots/_epoch_4.pkl")
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(67)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (width, height))

    # # Old cv2
    # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

    txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

    frame_num = 1
    count=0
    f=open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt','r')
    g=open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_pose_list.txt','r')
    while True:
            # get video frame
            ret, image = capture.read()
            image = cv2.flip(image,1)

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box of the face along with the associated
                    # probability

                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    img = image[startY:endY + 1, startX:endX + 1]





                    #x_size = crop_img.shape[1]
                    #y_size = crop_img.shape[0]
                    #img = crop_img[0:int(y_size*2/3),0:x_size]
                    #crop_img = crop_img[0:int(crop_img.shape[0]*2/3),0:crop_img.shape[1]]

                    img = Image.fromarray(img).convert('RGB')
                    img = transformations(img).float()
                    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])





                    #img = Variable(img).cuda(gpu)
                    #img.type(torch.float)

                    # Transform


                    yaw, pitch, roll = model(img.cuda())




                    yaw_predicted = nn.Softmax().cuda(gpu)(yaw)
                    pitch_predicted = nn.Softmax().cuda(gpu)(pitch)
                    roll_predicted = nn.Softmax().cuda(gpu)(roll)
                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor,1) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor,1) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data * idx_tensor,1) * 3 - 99


                    # Print new frame with cube and axis
                    print(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                    utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx=(startX + endX) / 2,
                                    tdy=(startY + endY) / 2, size=(endY-startY) / 2)
                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                    cv2.imshow("image",image)
                    cv2.waitKey(1)

            #out.write(image)

    #out.release()
    #video.release()
