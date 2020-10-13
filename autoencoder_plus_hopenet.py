import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dataset_autoencoder_recovery
import cv2
import torchvision
import math
from torch.autograd import Variable
import datasets_estimation_model
import datasets
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = dataset_autoencoder_recovery.Pose_300W_LP('/home/leechanhyuk/Downloads/NEW_IMAGE/','/home/leechanhyuk/Downloads/term_project/deep-head-pose-master/code/file_name_list.txt' ,transform,test=0)
test_data = dataset_autoencoder_recovery.Pose_300W_LP('/home/leechanhyuk/Downloads/NEW_IMAGE/','/home/leechanhyuk/Downloads/term_project/deep-head-pose-master/code/file_name_list_test.txt',transform,test=1)

estimation_train_data = datasets_estimation_model.Pose_300W_LP('/home/leechanhyuk/Downloads/mask1/abc','/home/leechanhyuk/PycharmProjects/tensorflow1/new_file_name_list.txt',transform,test=0)
estimation_test_data = datasets_estimation_model.Pose_300W_LP('/home/leechanhyuk/Downloads/mask1/abc','/home/leechanhyuk/PycharmProjects/tensorflow1/new_file_name_list.txt',transform,test=1)

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 64

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
estimation_train_loader = torch.utils.data.DataLoader(estimation_train_data,batch_size=batch_size,num_workers=num_workers)
estimation_test_loader = torch.utils.data.DataLoader(estimation_test_data,batch_size=batch_size,num_workers=num_workers)

pose_dataset = datasets.Pose_300W_LP('/home/leechanhyuk/Downloads/NEW_IMAGE','/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt' ,transform)
train_loader_no_mask = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=20, # 원래는 arg_parse로 받는 수 였음(수정함)
                                               shuffle=True,
                                               num_workers=2)

import matplotlib.pyplot as plt


# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


import torch.nn as nn
import torch.nn.functional as F
def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param
def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
# define the NN architecture
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
        decoder_output = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv3(x))

        return x


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
        self.avgpool = nn.AvgPool2d(4)

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


# initialize the NN
model = ConvAutoencoder(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3]).cuda(0)

print(model)

# specify loss function
criterion = nn.BCELoss().cuda(0)

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 100
gpu=0



for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for label,data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images = data.cuda()
        labels = label.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images.cuda())
        # calculate the loss
        loss = criterion(outputs, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * images.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))
    if epoch % 1 == 0:
        print('Taking snapshot...')
        torch.save(model.state_dict(),
                   'output/snapshots/' + '_epoch_' + str(epoch + 1) + '.pkl')
model.load_state_dict(torch.load('/home/leechanhyuk/Desktop/weights/Autoencoder/resnet_p1,p2,p3_added/_epoch_101.pkl'))
model.eval()
hopenet = Hopenet(torchvision.models.resnet.Bottleneck , [3, 4, 6, 3] , 67).cuda(gpu)
estimation_criterion = nn.CrossEntropyLoss().cuda(gpu)
reg_criterion = nn.MSELoss().cuda(gpu)
# Regression loss coefficient
softmax = nn.Softmax().cuda(gpu)
idx_tensor = [idx for idx in range(67)]
idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
estimation_optimizer = torch.optim.Adam([{'params': get_ignored_params(hopenet), 'lr': 0},
                                  {'params': get_non_ignored_params(hopenet), 'lr': 0.00001},
                                  {'params': get_fc_params(hopenet), 'lr': 0.00005}],
                                   lr = 0.001)
num_epochs=100
for epoch in range(100):
    for i, (images, labels, cont_labels, name) in enumerate(train_loader_no_mask):
        images = images.cuda()
        output = model(images)


        # use detach when it's an output that requires_grad
        # Binned labels
        label_yaw = Variable(labels[:, 0]).cuda(gpu)
        label_pitch = Variable(labels[:, 1]).cuda(gpu)
        label_roll = Variable(labels[:, 2]).cuda(gpu)
        # Continuous labels
        label_yaw_cont = Variable(cont_labels[:, 0]).cuda(gpu)
        label_pitch_cont = Variable(cont_labels[:, 1]).cuda(gpu)
        label_roll_cont = Variable(cont_labels[:, 2]).cuda(gpu)
        # Forward pass
        yaw, pitch, roll = (hopenet(output.cuda()))
        # Cross entropy loss
        loss_yaw = estimation_criterion(yaw, label_yaw)
        loss_pitch = estimation_criterion(pitch, label_pitch)
        loss_roll = estimation_criterion(roll, label_roll)
        # MSE loss
        yaw_predicted = softmax(yaw)
        pitch_predicted = softmax(pitch)
        roll_predicted = softmax(roll)
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99
        loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
        loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
        loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)
        # Total loss
        loss_yaw += 0.001 * loss_reg_yaw
        loss_pitch += 0.001 * loss_reg_pitch
        loss_roll += 0.001 * loss_reg_roll
        loss_seq = [loss_yaw, loss_pitch, loss_roll]
        # grad_seq = [torch.tensor(1,dtype=torch.float).cuda(gpu) for _ in range(len(loss_seq))]
        grad_seq = [torch.tensor(1, dtype=torch.float).cuda(gpu) for _ in range(len(loss_seq))]
        estimation_optimizer.zero_grad()
        torch.autograd.backward(loss_seq)
        estimation_optimizer.step()
        try:
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] '
                      % (epoch + 1, num_epochs, i + 1, len(train_loader_no_mask) // batch_size))
                print("a")
                print(loss_yaw)
                print(loss_pitch)
                print(loss_roll)
        except:

            print("b")
            print(loss_yaw)
    if epoch % 1 == 0 and epoch < num_epochs:
        print('Taking snapshot...')
        torch.save(hopenet.state_dict(),
                   '/home/leechanhyuk/Desktop/weights/20201012/' + '_epoch_' + str(epoch + 1) + '.pkl')


