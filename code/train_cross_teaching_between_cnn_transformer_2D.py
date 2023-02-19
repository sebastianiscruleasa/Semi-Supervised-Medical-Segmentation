# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the validation set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the validation set and test set to report the results.
#   We found that the random data split has some bias (the validation set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model1 = create_model()
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).to(device)
    #model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    if device == "cuda":
        train_num_workers = 4
        val_num_workers = 1
        pin_memory = True
        worker_init_function = worker_init_fn
    else:
        train_num_workers = 0
        val_num_workers = 0
        pin_memory = False
        worker_init_function = None

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=train_num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_function)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=val_num_workers)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0

    epoch = 0
    model1_loss = 0
    model2_loss = 0

    if config.MODEL.PRETRAIN_CKPT_MODEL1 is not None:
        loaded_model1 = torch.load(config.MODEL.PRETRAIN_CKPT_MODEL1)
        epoch = loaded_model1["epoch"] + 1
        iter_num = loaded_model1["iter"]
        model1.load_state_dict(loaded_model1["model"])
        optimizer1.load_state_dict(loaded_model1["optimizer"])
        best_performance1 = loaded_model1["best_performance1"]

    if config.MODEL.PRETRAIN_CKPT_MODEL2 is not None:
        loaded_model2 = torch.load(config.MODEL.PRETRAIN_CKPT_MODEL2)
        model2.load_state_dict(loaded_model2["model"])
        optimizer2.load_state_dict(loaded_model2["optimizer"])
        best_performance2 = loaded_model2["best_performance2"]

    iterator = tqdm(range(epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)

            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))

            if iter_num >= max_iterations:
                break
            time1 = time.time()

        # adding image data to tensorboard summary

        # code commented for now until I figure out what is with the * 50, whether is related to the fact
        # that this was done before for every 50 iterations
        # image = volume_batch[1, 0:1, :, :]
        # writer.add_image('train/Image', image, epoch_num)
        # outputs = torch.argmax(torch.softmax(
        #     outputs1, dim=1), dim=1, keepdim=True)
        # writer.add_image('train/model1_Prediction',
        #                  outputs[1, ...] * 50, epoch_num)
        # outputs = torch.argmax(torch.softmax(
        #     outputs2, dim=1), dim=1, keepdim=True)
        # writer.add_image('train/model2_Prediction',
        #                  outputs[1, ...] * 50, epoch_num)

        #saving checkpoints after every epoch and updating best model if neccesary
        model1.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(valloader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_val)
        for class_i in range(num_classes - 1):
            writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                              metric_list[class_i, 0], epoch_num)
            writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                              metric_list[class_i, 1], epoch_num)

        performance1 = np.mean(metric_list, axis=0)[0]

        mean_hd951 = np.mean(metric_list, axis=0)[1]
        writer.add_scalar('info/model1_val_mean_dice',
                          performance1, epoch_num)
        writer.add_scalar('info/model1_val_mean_hd95',
                          mean_hd951, epoch_num)

        drive_snapshot_path = "/content/gdrive/MyDrive/Licenta/Semi_Supervised_Medical_Segmentation_Checkpoints"
        save_mode_path = os.path.join(drive_snapshot_path,
                                      'epochs/model1_epoch_{}_dice_{}.pth'.format(
                                          epoch_num, round(best_performance1, 4)))
        if epoch_num % 300 == 0:
            torch.save({
                'model': model1.state_dict(),
                'optimizer': optimizer1.state_dict(),
                'epoch': epoch_num,
                'loss': model1_loss,
                'iter': iter_num,
                'best_performance1': performance1 if performance1 > best_performance1 else best_performance1
            }, save_mode_path)

        if performance1 > best_performance1:
            best_performance1 = performance1
            save_best = os.path.join(drive_snapshot_path,
                                     '{}_best_model1.pth'.format(args.model))
            torch.save({
                'model': model1.state_dict(),
                'optimizer': optimizer1.state_dict(),
                'epoch': epoch_num,
                'loss': model1_loss,
                'iter': iter_num,
            }, save_best)
            logging.info(
                'epoch %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (epoch_num, performance1, mean_hd951))

        model1.train()

        model2.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(valloader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_val)
        for class_i in range(num_classes - 1):
            writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                              metric_list[class_i, 0], epoch_num)
            writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                              metric_list[class_i, 1], epoch_num)

        performance2 = np.mean(metric_list, axis=0)[0]

        mean_hd952 = np.mean(metric_list, axis=0)[1]
        writer.add_scalar('info/model2_val_mean_dice',
                          performance2, epoch_num)
        writer.add_scalar('info/model2_val_mean_hd95',
                          mean_hd952, epoch_num)

        save_mode_path = os.path.join(drive_snapshot_path,
                                      'epochs/model2_epoch_{}_dice_{}.pth'.format(
                                          epoch_num, round(best_performance2, 4)))
        if epoch_num % 300 == 0:
            torch.save({
                'model': model2.state_dict(),
                'optimizer': optimizer2.state_dict(),
                'loss': model2_loss,
                'best_performance2': performance2 if performance2 > best_performance2 else best_performance2
            }, save_mode_path)

        if performance2 > best_performance2:
            best_performance2 = performance2
            save_best = os.path.join(drive_snapshot_path,
                                     '{}_best_model2.pth'.format(args.model))
            torch.save({
                'model': model2.state_dict(),
                'optimizer': optimizer2.state_dict(),
                'loss': model2_loss
            }, save_best)
            logging.info(
                'epoch %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (epoch_num, performance2, mean_hd952))

        model2.train()

        if epoch_num % 300 == 0:
            for filename in os.listdir(drive_snapshot_path + "/epochs"):
                if filename.find('_epoch_' + str(epoch_num)) == -1:
                    file_path = os.path.join(drive_snapshot_path + "/epochs", filename)
                    open(file_path, 'w').close()
                    os.remove(file_path)

        if iter_num >= max_iterations:
            iterator.close()
            break
        labs = label_batch[1, ...].unsqueeze(0) * 50
        writer.add_image('train/GroundTruth', labs, epoch_num)
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
