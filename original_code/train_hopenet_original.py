import sys, os, argparse, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import datasets, hopenet
import torch.utils.model_zoo as model_zoo
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    args = parser.parse_args()
    return args
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
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
def omp(D, data, sparsity):
    max_coeff = sparsity
    sparse_coeff = torch.empty(D.shape[1],data.shape[1], dtype=torch.float)
    atoms_list=[]
    for i in range(data.shape[1]):
        count = torch.floor((i+1)/data.shape[1]*100)
        x = data[:,i]
        res = x
        res_norm = res.norm()
        while len(atoms_list) < max_coeff:
           proj = torch.dot(torch.transpose(D),res)
           i_0 = torch.argmax(torch.abs(proj))
           atoms_list.append(i_0)
           temp_sparse = torch.dot(torch.pinverse(D[:,atoms_list]),x)
           res = x - torch.dot(D[:,atoms_list],temp_sparse)
           res_norm = res.norm()
        tot_res += res_norm
        if len(atoms_list) > 0:
           sparse_coeff[atoms_list,i] = temp_sparse
    return sparse_coeff
if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True # cudnn 승인
    num_epochs = 30 # 에폭
    batch_size = args.batch_size # 베치사이즈
    gpu = args.gpu_id # GPU ID 알아보기.
    if not os.path.exists('output/snapshots'): # snapshots directory 만들기
        os.makedirs('output/snapshots')
    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 67)
    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)
    print('Loading data.')
    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pose_dataset = datasets.Pose_300W_LP('/home/leechanhyuk/Downloads/NEW_IMAGE','/home/leechanhyuk/PycharmProjects/tensorflow1/file_name_list.txt' ,transformations)
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=20, # 원래는 arg_parse로 받는 수 였음(수정함)
                                               shuffle=True,
                                               num_workers=2)
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha
    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = [idx for idx in range(67)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': 0.00001},
                                  {'params': get_fc_params(model), 'lr': 0.00005}],
                                   lr = args.lr)
    #dictionary = np.load('file.npy')
    print('Ready to train network.')
    for epoch in range(30):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)
            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)
            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)
            # Forward pass
            yaw, pitch, roll = model(images)
            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)
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
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll
            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            #grad_seq = [torch.tensor(1,dtype=torch.float).cuda(gpu) for _ in range(len(loss_seq))]
            grad_seq = [torch.tensor(1,dtype=torch.float).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq)
            optimizer.step()
            try:
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] '
                        %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size))
                    print("a")
                    print(loss_yaw)
                    print(loss_pitch)
                    print(loss_roll)
            except:
                print("b")
                print(loss_yaw)
        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')
