# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import numpy as np
import random
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from callbacks import AverageMeter
from data_utils.causal_data_loader_frames import VideoFolder
from utils import save_results
from tqdm import tqdm

from model.model_lib import RobertaForMultipleChoiceWithLM
from model.GraphUtils import GraphUtils

from transformers import RobertaTokenizer, RobertaConfig

parser = argparse.ArgumentParser(description='KCMM')

# Path, dataset and log related arguments
parser.add_argument('--root_frames', type=str, default='/mnt/data1/home/sunpengzhan/sth-sth-v2/',
                    help='path to the folder with frames')
parser.add_argument('--json_data_train', type=str, default='../data/dataset_splits/compositional/train.json',
                    help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', type=str, default='../data/dataset_splits/compositional/validation.json',
                    help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', type=str, default='../data/dataset_splits/compositional/labels.json',
                    help='path to the json file with ground truth labels')

parser.add_argument('--dataset', default='smth_smth',
                    help='which dataset to train')
parser.add_argument('--logname', default='my_method',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')

# model, image&feature dim and training related arguments
parser.add_argument('--model_vision', default='rgb_roi')

parser.add_argument('--img_feature_dim', default=512, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=512, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=16, type=int,
                    help='num of frames for the model')
# 默认174
# base 88
# fewshot 86
# sthcom 161
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10, 15, 20], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--search_stride', type=int, default=5, help='test performance every n strides')

# train mode, hardware setting and others related arguments
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--cf_inference_group', action='store_true',
                    help='counterfactual inference model on validation set')
parser.add_argument('--parallel', default=True, type=bool,
                    help='whether or not train with multi GPUs')
# parser.add_argument('--gpu_index', type=str, default='0, 1', help='the index of gpu you want to use')
parser.add_argument('--best_acc', default=-1, type=float, help='best_acc')

parser.add_argument('--w_vision', default=1, type=float, help='w_vision')
parser.add_argument('--w_com', default=0.001, type=float, help='w_com')

best_loss = 1000000


def main():
    global args, best_loss
    args = parser.parse_args()
    print(args)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    config = RobertaConfig.from_pretrained('roberta-large')

    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    common_model = RobertaForMultipleChoiceWithLM.from_pretrained(
        '../code/model/pretrained_weights/roberta-large_model.bin',
        config=config)

    from model.model_lib import BboxVisualModel as RGBModel
    print('rgb_roi loaded!!')

    graph = GraphUtils()
    print('graph init...')
    graph.load_mp_all_by_pickle('../code/model/conceptnet5/res_all.pickle')
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()
    print('reduce graph noise done!')

    # load model branch
    vision_model = RGBModel(args)

    if args.parallel:
        vision_model = torch.nn.DataParallel(vision_model).cuda()

    else:
        vision_model = vision_model.cuda()

    common_model = torch.nn.DataParallel(common_model, device_ids=[1]).cuda()
    best_model_temp_path = '../code/model/pretrained_weights/common_best_model.pth'
    state_dict = torch.load(best_model_temp_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    common_model.load_state_dict(new_state_dict)

    for p in common_model.parameters():
        p.requires_grad = False

    common_model.eval()

    if args.start_epoch is None:
        args.start_epoch = 0

    cudnn.benchmark = True

    # create training and validation dataset
    dataset_train = VideoFolder(root=args.root_frames,
                                num_boxes=args.num_boxes,
                                file_input=args.json_data_train,
                                file_labels=args.json_file_labels,
                                frames_duration=args.num_frames,
                                args=args,
                                is_val=False,
                                if_augment=True,
                                )
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              )

    # create training and validation loader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    model_list = [vision_model, common_model, tokenizer, graph]

    optimizer_vision = torch.optim.SGD(filter(lambda p: p.requires_grad, vision_model.parameters()),
                                       momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)

    optimizer_list = [optimizer_vision]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=1000)

    if args.evaluate:
        validate(val_loader, model_list, criterion, class_to_idx=dataset_val.classes_dict)
        return

    print('training begin...')
    for epoch in tqdm(range(args.start_epoch, args.epochs)):

        adjust_learning_rate(optimizer_vision, epoch, args.lr_steps, 'vision')

        # train for one epoch
        train(train_loader, model_list, optimizer_list, epoch, criterion)

        if (epoch + 1) % 1 == 0:

            top1_acc_val = validate(val_loader, model_list, criterion,
                                    epoch=epoch, class_to_idx=dataset_val.classes_dict)

            if top1_acc_val > args.best_acc:
                is_best = top1_acc_val > args.best_acc
                args.best_acc = top1_acc_val
                if not os.path.exists(os.path.join(args.ckpt)):
                    os.makedirs(args.ckpt)
                print('save checkpoint bestacc:{}'.format(args.best_acc))

                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': vision_model.state_dict(),
                        'best_acc': args.best_acc,
                    },
                    is_best,
                    os.path.join(args.ckpt,
                                 '{}_{}_{}_{}'.format(args.model_vision, args.logname, epoch, args.best_acc)))


def train(train_loader, model_list,
          optimizer_list, epoch, criterion):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # losses = AverageMeter()
    losses_vision = AverageMeter()
    com_losses_vision = AverageMeter()

    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    [vision_model, common_model, tokenizer, graph] = model_list

    # load four optimizers, including the one designed for uniform assumption
    [optimizer_vision] = optimizer_list

    # switch to train mode
    vision_model.train()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label, gt_placeholders, verb_label,
            gt_placeholders_label_new_id) in enumerate(train_loader):

        data_time.update(time.time() - end)

        output_vision, feature_vision, output_com, feature_com, is_common_true_feature, is_common_false_feature, \
            true_label = vision_model(
            global_img_tensors.cuda(), box_tensors.cuda(), box_categories.cuda(),
            video_label, gt_placeholders, verb_label,
            gt_placeholders_label_new_id,
            common_model, tokenizer, graph)

        output_vision = output_vision.view((-1, len(train_loader.dataset.classes)))

        loss_vision = criterion(output_vision, video_label.long().cuda())

        com_loss_vision = criterion(feature_com, true_label.long().cuda())

        if torch.isnan(com_loss_vision).any():
            com_loss_vision = torch.tensor(0).cuda()

        # Measure the accuracy of the sum of three branch activation results
        acc1, acc5 = accuracy(output_vision.cpu(), video_label, topk=(1, 5))

        # record the accuracy and loss

        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))

        # refresh the optimizer
        optimizer_vision.zero_grad()

        losses_vision.update(loss_vision.item(), global_img_tensors.size(0))
        com_losses_vision.update(com_loss_vision.item(), global_img_tensors.size(0))

        loss = args.w_vision * loss_vision + args.w_com * com_loss_vision
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(vision_model.parameters(), args.clip_gradient)

        # update the parameter
        optimizer_vision.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Com_Loss {com_loss.val:.4f} ({com_loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses_vision, com_loss=com_losses_vision,
                acc_top1=acc_top1, acc_top5=acc_top5))


def validate(val_loader, model_list, criterion,
             epoch=None, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    logits_matrix = []
    targets_list = []

    [vision_model, common_model, tokenizer, graph] = model_list

    # switch to evaluate mode
    vision_model.eval()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label, gt_placeholders,
            verb_label,
            gt_placeholders_label_new_id) in enumerate(val_loader):
        # compute output
        with (torch.no_grad()):

            output_vision, feature_vision = vision_model(
                global_img_tensors.cuda(), box_tensors.cuda(), box_categories.cuda(),
                video_label, gt_placeholders, verb_label,
                gt_placeholders_label_new_id,
                common_model, tokenizer, graph, True)

            output_vision = output_vision.view((-1, len(val_loader.dataset.classes)))

            loss_vision = criterion(output_vision, video_label.long().cuda())

            output = output_vision
            loss = loss_vision

            acc1, acc5 = accuracy(output.cpu(), video_label, topk=(1, 5))

            if args.evaluate:
                logits_matrix.append(output.cpu().data.numpy())
                targets_list.append(video_label.cpu().numpy())

        # measure accuracy and record loss
        losses.update(loss.item(), global_img_tensors.size(0))
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                acc_top1=acc_top1, acc_top5=acc_top5,
            ))

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)
        save_results(logits_matrix, targets_list, class_to_idx, args)

    return acc_top1.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_steps, branch_name=None):
    """Sets the learning rate to the initial LR decayed by 10"""

    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    if branch_name == 'vision':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.8
    elif branch_name == 'coord':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif branch_name == 'fusion':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
