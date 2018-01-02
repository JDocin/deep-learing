# -*- coding:utf-8 -*-
"""
@author:Docin
@file:trainvgg.py
@time:2017/11/1517:09
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
#字典{图片名：[图片路径，标记路径，图片分片，坏块个数,[坏块坐标,修补坐标]

color = [0,0,0]
pic = {} #图片信息结构体
bbkcount = 0
good = [] #好块  0
bbk = [] #坏块   1
brp = [] #修补块 2
imgs = []
tagfiles = {} #存放标签

#读取标签文件函数
def tagfile(p):
    brplist = []
    bbklist = []
    fp = open(p)
    for line in fp:

        if line[0:15]=="Bad_BlockCount:":
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
    #print(bbklist)
    fp.close()
    return [bbklist,brplist]


#获取图片名称，路径，标签文件路径。
def readfile():
    FindPath = '/home/xgs/demo/ptvgg/testtrain'
    FileNames = os.listdir(FindPath)
    imgname = ''
    imgpath = ''
    # tagfile = ''
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
    #                 elif file2.endswith('.txt'):
    #                     tagfile = os.path.join(fullfilename1, file2)
    #             pic[imgname] = [imgpath, tagfile]
    #             imgname = imgpath = tagfile = ''

    for file_name in FileNames:
        fullfilename0 = os.path.join(FindPath, file_name)
        if fullfilename0.endswith('.txt'):
            if(fullfilename0[-13:] =='fileindex.txt'):
                for line in open(fullfilename0):
                    s = line.split('->')
                    s[2] = s[2][:-1]
                    a = s[2].replace('\\','/')

                    f = a+'/'+s[1][:-4]+'.txt'
                    tagfiles[s[1]] = FindPath+f

        else:
            for file1 in os.listdir(fullfilename0):
                fullfilename1 = os.path.join(fullfilename0, file1)
                for file2 in os.listdir(fullfilename1):
                    if file2.endswith('.jpg'):
                        imgname = file2
                        imgpath = os.path.join(fullfilename1, file2)
                pic[imgname] = [imgpath]
                imgname = imgpath = ''
    for i,j in pic.items():
        f = tagfiles.get(i)
        if (f):
            j.append(f)
            print(j)
readfile()


#切分图片，并且把碎片分类
def cutandleb():
    for name,path2 in pic.items():

        a = path2[0].replace('\\\\','/')
        img = cv2.imread(a)
        li = tagfile(path2[1])
        #print(li)
        path2.append(bbkcount)
        path2.append(li)
        #裁剪
        sh = img.shape
        y = sh[0]
        x = sh[1]
        i = y//100 #有几行
        j = x//100 #有几列
        listi = [] #行
        listj = [] #列

        for m in range(i) :
            del listj[:]
            for n in range(j) :
                img0 = img[m*100:100+m*100,n*100:100+n*100]
                try:
                    img0 = cv2.resize(img0, (224, 224), interpolation=cv2.INTER_CUBIC)
                except:
                    print('wrong')
                listj.append(img0)
            listi.append(listj)
        path2.append(listi)
        p = 0

        #统计坐标
        block = [0,0]
        for i in listi: #y为1
            q = 0
            p+=1
            block[0]=p
            for j in i: #x变化
                q+=1
                block[1]=q
                if block in li[0]:

                    # print("badblock",block)
                    # cv2.imshow("jay", j)
                    # cv2.waitKey(0)
                    bbk.append(j)
                elif block in li[1]:
                    # print("badrep",block)
                    brp.append(j)
                else:
                    good.append(j)
                #
                #print(block)
        del li[:]

cutandleb()
# good.extend(bbk)
# good.extend(brp)
# imgs = good
# print(bbk)

transform =transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]) #预处理

def default_loader(img):
    return Image.open(img)


class PicADD(data.Dataset):
    def __init__(self, good, bbk, brp, train=True, img_transform=None, loader=default_loader):
        self.label_list = []
        for i in good:
            self.label_list.append('0')
        for i in bbk:
            self.label_list.append('1')
        for i in brp:
            self.label_list.append('2')

        self.img_list = good+bbk+brp
        self.img_transform = img_transform
        self.loader = loader
        self.train = train  # training set or test set

    def __getitem__(self, index):
        img = self.img_list[index]
        label = self.label_list[index]
        # range [0, 255] -> [0.0,1.0]

        if self.img_transform is not None:
            img = self.img_transform(img)

        # label = list(label)
        # label = np.array(label)
        # label = label.astype('uint8')
        # label= torch.from_numpy(label)
        # label = label.type(torch.LongTensor)

        return img, label

    def __len__(self):
        return len(self.label_list)


picp = PicADD(good ,bbk ,brp,img_transform=transform)
trainloader = torch.utils.data.DataLoader(picp, batch_size=4,
                                          shuffle=True) #重新排列



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear(25088, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pool2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pool2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, 2, stride=2)

        h = h.view(-1, 25088)  # 由最后一层卷积定义的输出

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        return F.softmax(h)
# net = VGG()
net1 = mo.vgg16()
net1.eval()
if(torch.cuda.is_available()): #GPU
    print("gpu up")
    net1.cuda()
criterion = nn.CrossEntropyLoss() #叉熵损失函数
optimizer = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1):  # 遍历数据集两次
    running_loss = 0.0
    # enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        labels = list(labels)
        labels = np.array(labels)
        labels = labels.astype('uint8')
        labels= torch.from_numpy(labels)
        labels = labels.type(torch.LongTensor)

        #label = label.type(torch.LongTensor)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        # zero the parameter gradients
        # 将参数的grad值初始化为0
        optimizer.zero_grad()
        # 我们网络中的每一个参数都是Variable类型，
        # 并且均是叶子节点，grad值必然会进行更新。
        # forward + backward + optimize
        outputs = net1(inputs)
        # 将output和labels使用叉熵计算损失

        loss = criterion(outputs,labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 用SGD更新参数

        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 200 == 199:  # 每10批打印一次
            print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

torch.save(net1,'./net1.pkl')
