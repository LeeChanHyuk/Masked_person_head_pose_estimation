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
from skimage.util.shape import view_as_windows
from functools import reduce
from operator import mul, sub
import math



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


class ConvAutoencoder(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv4 = nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        #self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        #self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(2048,1024,2,stride=2)
        self.t_conv4 = nn.ConvTranspose2d(1024,256,2,stride=2)
        self.t_conv5 = nn.ConvTranspose2d(256,64,2,stride=2)
        self.t_conv6 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.t_conv7 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.conv2 = nn.Conv2d(16,4,3,padding=1)
        self.conv3 = nn.Conv2d(4, 3, 3, padding=1)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        #x = F.relu(self.conv1(x))
        #x = self.pool(x)
        # add second hidden layer
        #x = F.relu(self.conv2(x))
        #x = self.pool(x)  # compressed representation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        p1 = self.conv5(x)
        x = self.layer2(x)
        p2 = self.conv4(x)
        x = self.layer3(x)
        p3 = x
        encoder_output = self.layer4(x)
        ## decode ##
        # add transpose conv layers, with relu activation function
        #x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        #x = F.sigmoid(self.t_conv2(x))
        x=F.relu(self.t_conv3(encoder_output) + p3)
        x=F.relu(self.t_conv4(x) + p2)
        x=F.relu(self.t_conv5(x) + p1)
        x=F.relu(self.t_conv6(x))
        x=F.relu(self.t_conv7(x))
        x = F.relu(self.conv2(x))
        decoder_output = F.sigmoid(self.conv3(x))

        return encoder_output

class pose_estimation(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(pose_estimation, self).__init__()
        self.avgpool = nn.AvgPool2d(4)
        self.fc_angles = nn.Linear(512 * block.expansion, num_bins)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw,pre_pitch,pre_roll
class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

gpu=0

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
    # initialize the NN
    model = ConvAutoencoder(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])

    estimation_model = pose_estimation(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 67)
    estimation_model.cuda(gpu)
    hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 67).cuda()
    hopenet.load_state_dict(torch.load('/home/leechanhyuk/Downloads/term_project/deep-head-pose-master/code/output/snapshots/before/LITTLE_SUCCESS_MASK/_epoch_30.pkl'))

    print('Loading snapshot.')
    # Load snapshot

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
    count=1
    f=open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt','r')
    g=open('/home/leechanhyuk/PycharmProjects/tensorflow1/file_pose_list.txt','r')
    x = 0
    y = 0
    average_x = 0
    average_y = 0
    transform = transforms.ToTensor()
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
                    img = cv2.resize(img,dsize=(224,224))
                    img = transformations(Image.fromarray(img))
                    img = img.reshape((1,3,224,224))






                    #x_size = crop_img.shape[1]
                    #y_size = crop_img.shape[0]
                    #img = crop_img[0:int(y_size*2/3),0:x_size]
                    #crop_img = crop_img[0:int(crop_img.shape[0]*2/3),0:crop_img.shape[1]]

                    #img = Image.fromarray(img).convert('RGB')
                    images = Variable(img).cuda(gpu)
                    yaw, pitch, roll = hopenet(images)




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
