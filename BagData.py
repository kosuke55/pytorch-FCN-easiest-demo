import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import sys
OPENCV_PATH \
    = '/home/kosuke/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages'
sys.path = [OPENCV_PATH] + sys.path
print(sys.path)
import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor()])


DATA_PATH \
    = "/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e/baidu_train_data/"


class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir(DATA_PATH + 'in_feature'))
        # return len(os.listdir('bag_data'))

    def __getitem__(self, idx):
        data_name = os.listdir(DATA_PATH + 'in_feature')[idx]
        in_feature = np.load(DATA_PATH + "in_feature/" + data_name)
        in_feature = in_feature[..., 7]
        # print(in_feature[in_feature > 0])
        in_feature = in_feature.astype(np.uint8)
        print(in_feature[in_feature > 0])
        # in_feature = in_feature * 255
        in_feature = cv2.resize(in_feature, (160, 160))
        in_feature = cv2.cvtColor(in_feature, cv2.COLOR_GRAY2RGB)

        # for check
        # in_feature = np.load(DATA_PATH + "out_feature/" + data_name)
        # in_feature = in_feature.astype(np.uint8)
        # in_feature = cv2.resize(in_feature, (160, 160))
        # in_feature = in_feature[..., 1]

        # loss_weight = np.load(DATA_PATH + "loss_weight/" + data_name)
        # loss_weight = loss_weight.astype(np.uint8)
        # loss_weight = cv2.resize(loss_weight, (160, 160))
        # loss_weight = cv2.cvtColor(loss_weight, cv2.COLOR_GRAY2RGB)
        # print(loss_weight.shape)
        out_feature = np.load(DATA_PATH + "out_feature/" + data_name)
        out_feature = out_feature.astype(np.uint8)
        out_feature = cv2.resize(out_feature, (160, 160))
        # print(out_feature.shape)
        out_feature = out_feature[..., 1]
        # print(out_feature.shape)

        out_feature = onehot(out_feature, 5)

        out_feature = out_feature.transpose(2, 0, 1)
        out_feature = torch.FloatTensor(out_feature)

        # img_name = os.listdir('bag_data')[idx]
        # imgA = cv2.imread('bag_data/'+img_name)
        # imgA = cv2.resize(imgA, (160, 160))
        # print(imgA.shape)
        # imgB = cv2.imread('bag_data_msk/'+img_name, 0)
        # imgB = cv2.resize(imgB, (160, 160))
        # print(imgB.shape)
        # imgB = imgB/255
        # imgB = imgB.astype('uint8')
        # imgB = onehot(imgB, 2)
        # imgB = imgB.transpose(2, 0, 1)
        # imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)

        # if self.transform:
        #     imgA = self.transform(imgA)
        in_feature = in_feature.astype(np.float32)
        print(in_feature[in_feature > 0])
        if self.transform:
            in_feature = self.transform(in_feature)
            print(in_feature[in_feature > 0])
        return in_feature, out_feature
        # return imgA, imgB


bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=1)


if __name__ == '__main__':

    for train_batch in train_dataloader:
        pass
        # print(train_batch)

    for test_batch in test_dataloader:
        pass
        # print(test_batch)
