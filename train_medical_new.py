import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from nets.unet import Unet, vgg_Unet_Attention, vgg_Unet_AF, Graph_network
from nets.unet_training import CE_Loss, Dice_loss
from utils.dataloader_medical1 import DeeplabDataset, deeplab_dataset_collate
from utils.dataloader_test import DeeplabDataset_test, deeplab_dataset_collate_test
from utils.metrics import f_score
from nets.unet_contrast import U_Net, AttU_Net, R2AttU_Net, R2U_Net
from utils.hausdorff import HausdorffDTLoss,HDLoss
from torchsummary import summary
# from unet import Unet
import torch.nn.functional as F
# from nets.ours_dattention import da_network
from nets.IIAM import da_network
from nets.IAM_NOIHAM import da_network_noIHAM
import csv

import os
# from thop import profile
# from thop import clever_format
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda,a):
    total_loss = 0
    total_f_score = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    total_ce=0
    total_dice=0
    total_hd=0

    val_total_loss = 0
    val_total_f_score = 0
    val_total_precision = 0
    val_total_recall = 0
    val_total_accuracy = 0
    val_total_ce = 0
    val_total_dice = 0
    val_total_hd = 0
    net = net.train()
    start_time = time.time()
    pbar: tqdm
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels, name = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(imgs)
            waste_time = time.time() - start_time
            loss_ = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = a*(loss_ + main_dice)
            if hausdorff:
                hd=HDLoss(pred_=outputs,target_=labels)
                bloss = (1 - a) * hd
                # bloss = (1 - a) * HDLoss(pred_=outputs, target_=labels)
                loss=loss+bloss

            with torch.no_grad():
                # -------------------------------#
                #   Calculate f_score
                # -------------------------------#
                (_f_score, precision, recall) = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            total_precision += precision.item()
            total_recall += recall.item()
            # total_accuracy += accuracy.item()
            total_accuracy=1
            total_ce +=loss_
            total_dice +=main_dice
            if hausdorff:
                total_hd += hd
            # waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'dice': total_f_score / (iteration + 1),
                                'precision': total_precision / (iteration + 1),
                                'recall': total_recall / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            # f = open(os.path.join('Result', 'result_train.csv'), 'a', encoding='utf-8', newline='')
            # wr = csv.writer(f)
            # wr.writerow(
            #     [epoch,total_loss / (iteration + 1), total_f_score / (iteration + 1),
            #                     total_precision / (iteration + 1),
            #                     total_recall / (iteration + 1),
            #                     get_lr(optimizer)])
            # f.close()
            pbar.update(1)

    start_time1 = time.time()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels, name = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs = net(imgs)
                val_loss_ = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    val_loss = a *(val_loss_ + main_dice)

                if hausdorff:
                    val_hd=HDLoss(pred_=outputs, target_=labels)
                    val_bloss = (1 - a)*val_hd

                    # val_bloss = (1 - a) * HDLoss(pred_=outputs, target_=labels)
                    val_loss = val_loss + val_bloss
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    # (_f_score, precision, recall, accuracy, fpr) = f_score(outputs, labels)
                    (_f_score, precision, recall) = f_score(outputs, labels)

                val_total_loss += val_loss.item()
                val_total_f_score += _f_score.item()
                val_total_precision += precision.item()
                val_total_recall += recall.item()
                # val_total_accuracy += accuracy.item()
                val_total_accuracy =1

                val_total_ce += val_loss_
                val_total_dice += main_dice
                if hausdorff:
                    val_total_hd += val_hd

            pbar.set_postfix(**{'val_total_loss': val_total_loss / (iteration + 1),
                                'val_dice': val_total_f_score / (iteration + 1),
                                'val_precision': val_total_precision / (iteration + 1),
                                'val_recall': val_total_recall / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1),
                                'val_lr': get_lr(optimizer)})
            # f = open(os.path.join('Result', 'result_val.csv'), 'a', encoding='utf-8', newline='')
            # wr = csv.writer(f)
            # wr.writerow(
            #     [epoch, val_total_loss / (iteration + 1), _f_score,val_total_f_score / (iteration + 1),
            #      val_total_precision / (iteration + 1),val_total_recall / (iteration + 1),
            #      get_lr(optimizer)])
            # f.close()
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print(
        'Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_total_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_total_loss / (epoch_size_val + 1)))


    # -------------------------------------------------------------------------------------------#
    #  Save training results
    # ---------------------------------------------------------------------------------------------#
    f = open(os.path.join('Result', 'result_train.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(
        [epoch + 1, total_loss / (epoch_size + 1), total_f_score / (epoch_size + 1),
         total_precision / (epoch_size + 1),
         total_recall / (epoch_size + 1),
         total_accuracy / (epoch_size + 1),
         total_ce/(epoch_size + 1),
         total_dice/(epoch_size + 1),
         total_hd/(epoch_size + 1)])
    f.close()
    f = open(os.path.join('Result', 'result_val.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(
        [epoch + 1, val_total_loss / (epoch_size_val + 1), val_total_f_score / (epoch_size_val + 1),
         val_total_precision / (epoch_size_val + 1), val_total_recall / (epoch_size_val + 1),
         val_total_accuracy / (epoch_size_val + 1),
         val_total_ce/(epoch_size_val + 1),
         val_total_dice/(epoch_size_val + 1),
         val_total_hd/(epoch_size_val + 1)])
    f.close()


# ---------------------------------------------------------------------------------------------
#                         Test
# ---------------------------------------------------------------------------------------------
# start_time1 = time.time()


def fit_test1(i, net, epoch, epoch_size, gentest, Epoch, cuda):
    total_loss = 0
    total_f_score = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    net = net.eval()
    model_path = './logs/log_f/iiam.pth'
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)
    print('{} model loaded.'.format(model_path))

    for iteration, batch in enumerate(gentest):
        imgs, pngs, labels, name = batch
        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
            pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
            labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
            if cuda:
                imgs = imgs.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()
            start_time1 = time.time()
            outputs = net(imgs)
            waste_time = time.time() - start_time1
            loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
            ##Output segmented images
            # n, c, h, w = outputs.size()
            # colors = [(255, 255, 255), (0, 0, 0)]
            # for i in range(n):
            #     pr = outputs[i]
            #     pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            #     # pr = pr[int((inputs_size[0] - h) // 2):int((inputs_size[0] - h) // 2 + h),
            #     #  int((inputs_size[1] - w) // 2):int((inputs_size[1] - w) // 2 + w)]
            #
            #     seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            #     for c in range(NUM_CLASSES):
            #         seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            #         seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            #         seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
            #     # image = Image.fromarray(np.uint8(seg_img)).resize((w, h))
            #     image = Image.fromarray(np.uint8(seg_img))
            #     image.save('Medical_Datasets/our_test_open/' + name[i] + '.jpg')
        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice
        with torch.no_grad():
            # (_f_score, precision, recall, accuracy, fpr) = f_score(outputs, labels)
            (_f_score, precision, recall) = f_score(outputs, labels)
        total_loss += loss.item()
        total_f_score += _f_score.item()
        total_precision += precision.item()
        total_recall += recall.item()
        # total_accuracy += accuracy.item()
        total_accuracy=1
        print(waste_time)
        print('[Test]loss: %.4f, Dice:%.4f, Precision: %.4f, Sensitive: %.4f' % (
            loss.item(), _f_score.item(), precision.item(), recall.item()))
    # waste_time = time.time() - start_time1
    print('[Test_average]loss: %.4f, Dice:%.4f, Precision: %.4f, Sensitive: %.4f' % (
        total_loss / epoch_size, total_f_score / epoch_size, total_precision / epoch_size, total_recall / epoch_size))
    # print(waste_time)
    f = open(os.path.join('Result', 'result_test.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(
        [i + 1, total_loss / epoch_size, total_f_score / epoch_size, total_precision / epoch_size,
         total_recall / epoch_size, total_accuracy / epoch_size])
    f.close()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    para_size = total_num * 4 / 1024 / 1024
    return {'Total': para_size, 'Trainable': trainable_num}

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == "__main__":
    log_dir = "./logs/"
    # ------------------------------#
    #   Training Validation Mode    Test Mode
    # ------------------------------#
    mode='train'
    # mode = 'test'
    # ------------------------------#
    #   the size of input image
    # ------------------------------#
    inputs_size = [512,512,3]
    # ---------------------#
    #   classification number+1
    #   Background + hepatic vessel
    # ---------------------#
    NUM_CLASSES = 2
    # --------------------------------------------------------------------#
    #Suggested
    #options:
    # Set to True when there are few types (several types)
    # When there are multiple types (more than ten), if batch_ If the size is relatively large (above 10), set it to True
    # When there are multiple types (more than ten), if batch_ If the size is relatively small (below 10), set it to False
    # ---------------------------------------------------------------------#
    dice_loss = True
    hausdorff = False
    # --------------------------------------#
    #   The use of pretraining weights for backbone networks
    #   need to train pretraining parameters? Starting from scratch training needs to be set to True
    # --------------------------------------#
    pretrained = False
    # -------------------------------#
    #   Use of Cuda
    # -------------------------------#
    Cuda = True

    # Obtain the model to train
    # model = U_Net(img_ch=3, output_ch=NUM_CLASSES).train()      # UNET
    # model=AttU_Net(img_ch=3, output_ch=NUM_CLASSES).train()     # ATTUNET
    # model = vgg_Unet_Attention(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()  # VGG_ATT_UNET
    # model = Graph_network(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train() # GAM+IAM
    # model = vgg_Unet_AF(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train() #VGGUNAT_AFF
    model = da_network(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train() #IIAM(SAM)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print(get_parameter_number(model))
    print(count_param(model))

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # --------------------------------------------------------------------#
    #                         vessel_segmentation
    # ---------------------------------------------------------------------#

    with open(r"./Medical_Datasets/ImageSets/Segmentation/train3.txt", "r") as f:
        train_lines = f.readlines()
    # Open the txt of the dataset
    with open(r"./Medical_Datasets/ImageSets/Segmentation/val.txt", "r") as f:
        val_lines = f.readlines()

    with open(r"./Medical_Datasets/ImageSets/Segmentation/test.txt", "r") as f:
        test_lines = f.readlines()

    # ------------------------------------------------------#
    #   The backbone feature extraction network features are universal, and frozen training can accelerate training speed
    #   It can also prevent weight damage during the early stages of training.
    #   Init_ Epoch is the starting iteration
    #   Interval_ Epoch is an iteration of frozen training
    #   Epoch Total Training Iteration
    #   Insufficient graphics memory, please reduce the Batch_size
    # ------------------------------------------------------#
    a = 1
    if mode == 'train':
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 50  # Temporarily not freezing training
        Batch_size = 3

        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, True)

        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        genval = DataLoader(val_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                            drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        # for param in model.vgg.parameters():
        #     param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = True


        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, genval, Interval_Epoch, Cuda,a)
            # a=a-0.01
            lr_scheduler.step()

    if mode == 'train':
        lr = 1e-5
        Interval_Epoch = 50
        Epoch = 100
        Batch_size = 3

        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, True)

        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        genval = DataLoader(val_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                            drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        # for param in model.vgg.parameters():
        #     param.requires_grad = True
        for param in model.parameters():
            param.requires_grad = True

        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, genval, Interval_Epoch, Cuda,a)
            # if epoch == 99:
            #     hausdorff=True
            # if epoch>=100:
            #     a=a-0.02
            lr_scheduler.step()

    # -------------------------------------------------------------------------------------------------
    #                                Test
    # -------------------------------------------------------------------------------------------------
    if mode == 'test':
        lr = 1e-5
        Interval_Epoch = 1
        Epoch = 1
        Batch_size = 3

        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        test_dataset = DeeplabDataset(test_lines, inputs_size, NUM_CLASSES, True)

        gentest = DataLoader(test_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)
        epoch_size = max(1, len(test_lines) // Batch_size)

        for i in range(10):
            fit_test1(i, model, Epoch, epoch_size, gentest, Interval_Epoch, Cuda)
        # lr_scheduler.step()
