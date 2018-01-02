# -*- coding:utf-8 -*-
"""
@author:Docin
@file:testvgg.py
@time:2017/11/238:12
"""
import torch.utils.data as data
from PIL import Image
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision.models as mo
import torchvision.transforms as transforms
import torch.optim as optim
import torch.tensor as te
import numpy as np
import numpy

testpic = {}
tlabel = []  # [['path',y,x,tag] , ['path',y,x]]
tpic = []
ttagfiles = {}  # 测试集的标签


def testtagfile(p):
    brplist = []
    bbklist = []
    fp = open(p)
    for line in fp:

        if line[0:15] == "Bad_BlockCount:":
            count = line[15:]

        elif line[0:13] == "Bad_BlockPos:":
            if line[14].isalnum():
                y = str(line[13]) + str(line[14])
                x = line[16:]
            else:
                y = line[13]
                x = line[15:]
            bbklist.append([int(y), int(x)])
        elif line[0:19] == "Bad_RepairBlockPos:":
            if line[20].isalnum():
                y = str(line[19]) + str(line[20])
                x = line[22:]
            else:
                y = line[19]
                x = line[21:]
            brplist.append([int(y), int(x)])
    # print(bbklist)
    fp.close()
    return [bbklist, brplist]


def testreadfile():
    FindPath = '/home/xgs/demo/ptvgg/testtrain'
    FileNames = os.listdir(FindPath)
    imgname = ''
    imgpath = ''

    # for file_name in FileNames:
    #     fullfilename0 = os.path.join(FindPath, file_name)
    #     if fullfilename0.endswith('.txt'):
    #         print("txt")
    #     else:
    #         for file1 in os.listdir(fullfilename0):
    #             fullfilename1 = os.path.join(fullfilename0, file1)
    #             for file2 in os.listdir(fullfilename1):
    #                 if file2.endswith('.jpg'):
    #                     imgname = file2
    #                     imgpath = os.path.join(fullfilename1, file2)
    #
    #             testpic[imgname] = [imgpath]
    #             imgname = imgpath  = ''

    for file_name in FileNames:
        fullfilename0 = os.path.join(FindPath, file_name)
        if fullfilename0.endswith('.txt'):
            if (fullfilename0[-13:] == 'fileindex.txt'):
                for line in open(fullfilename0):
                    s = line.split('->')
                    s[2] = s[2][:-1]
                    a = s[2].replace('\\', '/')
                    f = a + '/' + s[1][:-4] + '.txt'
                    ttagfiles[s[1]] = FindPath + f

        else:
            for file1 in os.listdir(fullfilename0):
                fullfilename1 = os.path.join(fullfilename0, file1)
                for file2 in os.listdir(fullfilename1):
                    if file2.endswith('.jpg'):
                        imgname = file2
                        imgpath = os.path.join(fullfilename1, file2)
                testpic[imgname] = [imgpath]
                imgname = imgpath = ''
    for i, j in testpic.items():
        f = ttagfiles.get(i)
        if (f):
            j.append(f)
            print(j)


testreadfile()


# 切分图片，并且把碎片分类
def testcutandleb():
    for name, path2 in testpic.items():

        a = path2[0].replace('\\', '/')
        print(a)
        b = path2[1].replace('\\', '/')
        img = cv2.imread(a)
        li = testtagfile(b)
        try:
            # img = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(a, img)
            sh = img.shape

            y = sh[0]  # 10行
            x = sh[1]  # 14列
            i = y // 100  # 有几行
            j = x // 100  # 有几列

            p = 0
            for m in range(i):
                p += 1
                q = 0
                for n in range(j):
                    path = ['', 0, 0, 0]
                    q += 1
                    img0 = img[m * 100:100 + m * 100, n * 100:100 + n * 100]
                    img0 = cv2.resize(img0, (224, 224), interpolation=cv2.INTER_CUBIC)

                    tpic.append(img0)
                    if [p, q] in li[0]:
                        tag = '1'
                    elif [p, q] in li[1]:
                        tag = '2'
                    else:
                        tag = '0'
                    path[0] = path2[0]
                    path[1] = p
                    path[2] = q
                    path[3] = tag
                    tlabel.append(path)
        except:
            print('wrong')


testcutandleb()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])  # 预处理


def default_loader(img):
    return Image.open(img)


class testPicADD(data.Dataset):
    def __init__(self, testp, train=False, img_transform=None, loader=default_loader):
        self.img_list = testp
        self.img_transform = img_transform
        self.loader = loader
        self.train = train  # training set or test set

    def __getitem__(self, index):
        img = self.img_list[index]
        # range [0, 255] -> [0.0,1.0]

        if self.img_transform is not None:
            img = self.img_transform(img)

        # label = list(label)
        # label = np.array(label)
        # label = label.astype('uint8')
        # label= torch.from_numpy(label)
        # label = label.type(torch.LongTensor)

        return img

    def __len__(self):
        return len(self.img_list)


testp = testPicADD(tpic, img_transform=transform)
testloader = torch.utils.data.DataLoader(testp, batch_size=4,
                                         shuffle=False)  # 重新排列

net = torch.load('net1.pkl')
net.eval()
k = 0
right = 0
for data1 in testloader:

    images = data1
    outputs = net(Variable(images).cuda())
    # print outputs.data
    _, predicted = torch.max(outputs.data, 1)  # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
    # print(predicted)
    j = 0
    predicted = predicted.cpu()
    pred = predicted.numpy()  # 转为数组
    labels = tlabel

    for i in range(4):
        a = int(labels[k + j][3])

        if (pred[j] == a):
            right += 1


        if (pred[j] == 1):
            print(pred[j])
            path = labels[k + j][0]
            path = path[:-3]
            path = path + 'txt'
            f = open(path, 'a')
            line = 'BadBlock' + ':(' + str(labels[k + j][1]) + ',' + str(labels[k + j][2]) + ')' + '\n'
            f.write(line)
            f.close()
            img = cv2.imread(labels[k + j][0])
            cv2.rectangle(img, (100 * labels[k + j][2] - 100, 100 * labels[k + j][1] - 100),
                          (100 * labels[k + j][2], 100 * labels[k + j][1]), (0, 255, 0),
                          2)  # (画矩形函数) 左上角顶点，右下角顶点，色彩，边框粗度
            cv2.imwrite(labels[k + j][0], img)
        elif (pred[j] == 2):
            path = labels[k + j][0]
            path = path[:-3]
            path = path + 'txt'
            f1 = open(path, 'a')
            line1 = 'BadBlockRepaire' + ':(' + str(labels[k + j][1]) + ',' + str(labels[k + j][2]) + ')' + '\n'
            f1.write(line1)
            f1.close()
            img1 = cv2.imread(labels[k + j][0])
            cv2.rectangle(img1, (100 * labels[k + j][2] - 100, 100 * labels[k + j][1] - 100),
                          (100 * labels[k + j][2], 100 * labels[k + j][1]), (255, 0, 0), 2)  # (画矩形函数) 左上角顶点，右下
            cv2.imwrite(labels[k + j][0], img1)
        j += 1
        if k % 100 == 96:  # 每100批打印一次
            print('[%d] right: %.8f' % (k, right / k))
            running_loss = 0.0
    k += 4