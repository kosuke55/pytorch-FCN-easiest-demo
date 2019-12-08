from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from BagData import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet


def train(epo_num=50, show_vgg_params=False):

    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=5)
    fcn_model = fcn_model.to(device)
    # criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])
            pos_weight = bag_msk.detach().numpy().copy()
            pos_weight = pos_weight[0]
            pos_weight = pos_weight.transpose(1, 2, 0)
            zeroidx = np.where(pos_weight[:, :, 0] == 0)
            nonzeroidx = np.where(pos_weight[:, :, 0] != 0)
            pos_weight[zeroidx] = 100
            pos_weight[nonzeroidx] = 0
            pos_weight = pos_weight[..., 0]
            pos_weight_img = pos_weight[:, :, None]
            pos_weight_img = pos_weight_img.transpose(2, 0, 1)
            pos_weight = torch.from_numpy(pos_weight)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
            bag = bag.to(device)

            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            # output.shape is torch.Size([4, 2, 160, 160])
            output = torch.sigmoid(output)

            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            # output_np.shape = (4, 2, 160, 160)
            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmax(output_np, axis=1)
            output_np = output_np.transpose(1, 2, 0)
            car_idx = np.where(output_np[:, :, 0] == 1)
            bus_idx = np.where(output_np[:, :, 0] == 2)
            bike_idx = np.where(output_np[:, :, 0] == 3)
            human_idx = np.where(output_np[:, :, 0] == 4)

            output_img = np.zeros((160, 160, 3))
            output_img[car_idx] = [255, 0, 0]
            output_img[bus_idx] = [0, 255, 0]
            output_img[bike_idx] = [0, 0, 255]
            output_img[human_idx] = [0, 255, 255]
            output_img = output_img.transpose(2, 0, 1)
            # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy()
            bag_msk_np = np.argmax(bag_msk_np, axis=1)
            # print(bag_msk_np.shape)

            # bag_msk_np = np.argmax(bag_msk_np, axis=4)
            bag_msk_np = bag_msk_np.transpose(1, 2, 0)
            # print(bag_msk_np[bag_msk_np > 0])
            car_idx = np.where(bag_msk_np[:, :, 0] == 1)
            bus_idx = np.where(bag_msk_np[:, :, 0] == 2)
            bike_idx = np.where(bag_msk_np[:, :, 0] == 3)
            human_idx = np.where(bag_msk_np[:, :, 0] == 4)

            label_img = np.zeros((160, 160, 3))
            label_img[car_idx] = [255, 0, 0]
            label_img[bus_idx] = [0, 255, 0]
            label_img[bike_idx] = [0, 0, 255]
            label_img[human_idx] = [0, 255, 255]
            label_img = label_img.transpose(2, 0, 1)


            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo,
                                                                index, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(output_img, win='train_pred', opts=dict(
                    title='train prediction'))
                vis.images(label_img,
                           win='train_label', opts=dict(title='label'))
                vis.images(pos_weight_img,
                               win='hoge', opts=dict(title='a'))
                vis.line(all_train_iter_loss, win='train_iter_loss',
                         opts=dict(title='train iter loss'))

            # plt.subplot(1, 2, 1)
            # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            # plt.pause(0.5)

        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                # output.shape is torch.Size([4, 2, 160, 160])
                output = torch.sigmoid(output)
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                # output_np.shape = (4, 2, 160, 160)
                output_np = output.cpu().detach().numpy().copy()
                # output_np = np.argmin(output_np, axis=1)

                output_np = np.argmax(output_np, axis=1)
                output_np = output_np.transpose(1, 2, 0)

                car_idx = np.where(output_np[:, :, 0] == 1)
                bus_idx = np.where(output_np[:, :, 0] == 2)
                bike_idx = np.where(output_np[:, :, 0] == 3)
                human_idx = np.where(output_np[:, :, 0] == 4)

                output_img = np.zeros((160, 160, 3))
                output_img[car_idx] = [255, 0, 0]
                output_img[bus_idx] = [0, 255, 0]
                output_img[bike_idx] = [0, 0, 255]
                output_img[human_idx] = [0, 255, 255]
                output_img = output_img.transpose(2, 0, 1)
                # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy()
                bag_msk_np = np.argmax(bag_msk_np, axis=1)
                # print(bag_msk_np.shape)

                # bag_msk_np = np.argmax(bag_msk_np, axis=4)
                bag_msk_np = bag_msk_np.transpose(1, 2, 0)
                # print(bag_msk_np[bag_msk_np > 0])
                car_idx = np.where(bag_msk_np[:, :, 0] == 1)
                bus_idx = np.where(bag_msk_np[:, :, 0] == 2)
                bike_idx = np.where(bag_msk_np[:, :, 0] == 3)
                human_idx = np.where(bag_msk_np[:, :, 0] == 4)

                label_img = np.zeros((160, 160, 3))
                label_img[car_idx] = [255, 0, 0]
                label_img[bus_idx] = [0, 255, 0]
                label_img[bike_idx] = [0, 0, 255]
                label_img[human_idx] = [0, 255, 255]
                label_img = label_img.transpose(2, 0, 1)


                # bag_msk_np.shape = (4, 2, 160, 160)
                # bag_msk_np = bag_msk.cpu().detach().numpy().copy()
                # bag_msk_np = np.argmin(bag_msk_np, axis=1)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_img, win='test_pred', opts=dict(
                        title='test prediction'))
                    vis.images(label_img,
                               win='test_label', opts=dict(title='label'))

                    vis.line(all_test_iter_loss, win='test_iter_loss',
                             opts=dict(title='test iter loss'))

                # plt.subplot(1, 2, 1)
                # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
                # plt.subplot(1, 2, 2)
                # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                # plt.pause(0.5)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
              % (train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))

        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))


if __name__ == "__main__":

    train(epo_num=10000, show_vgg_params=False)
