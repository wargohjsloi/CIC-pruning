import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import sys
import copy
from tqdm import tqdm
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from models import resnet_cifar, resnet_imagenet
from autoaug import CIFAR10Policy, Cutout
from dataset.imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
from dataset.imbalance_cifar_moco import ImbalanceCIFAR10_MOCO, ImbalanceCIFAR100_MOCO
import torchvision.datasets as datasets
from torchvision.models import resnet
from utils import shot_acc
from losses import GMLLoss, DistillKL
from models.normed_linear import NormedLinear
import moco.loader
import moco.builder

from taylor_step_pruner import TaylorStepPruner
from taylor_merge_pruner import TaylorMergePruner
from fpgm_pruner import FPGMImportance, FPGMPruner
from cic_pruner import CICPruner
from clsaware_pruner import ClassAwarePruner
from base_train import set_dataset_imagenet, set_dataset_cifar, ProgressMeter, \
    str2bool, adjust_learning_rate, accuracy, save_checkpoint, AverageMeter
import torch_pruning as tp


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model_names += ['resnext101_32x4d']
model_names += ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', choices=['inat', 'imagenet', 'cifar100', 'cifar10'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./outputs')
parser.add_argument('--eval-log', type=str, default='./outputs/eval_out/test.txt')
parser.add_argument('--pretrained', type=str, default='./outputs/cifar100/ce_CIFAR100_imb010_R32_f4_B_lr0.1_0/ckpt.best.pth.tar')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--val-freq', default=2, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--imb-factor', type=float, default=0.10,
                    metavar='IF', help='imbalanced factor', dest='imb_factor')
parser.add_argument('--lr', '--learning-rate', default=0.10, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default=None, type=str,
                    help='resume the pruned model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda', type=str,
                    help='device for using.')
# pruning configs
parser.add_argument('--imp', default='l1', type=str,
                    choices=['l1', 'l2', 'taylor', 'taylor_v2', 'hessian', 'random', 'bns', 'fpgm', 'merge', 'cic', 'cls-aware'],
                    help='importance criterion for pruning. supported: l1, l2, taylor, hessian')
parser.add_argument('--pruning-ratio', '--pr', default=0.3, type=float,
                    help='pruned ratio for model parameter.')
parser.add_argument('--global', dest='global_pruning', action='store_true',
                    help='whether to use global pruning')
parser.add_argument('--iterative_steps', default=1, type=int,
                    help='iterative pruning steps')
parser.add_argument('--num-type', type=str, default='all', choices=['all', 'many', 'mid', 'few'],
                    help='type of data num for importance estimation')
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--cic_batch', type=int, default=128,  help='batch_size for re-organize dataset with padding')
parser.add_argument('--cic-weight', type=float, default=1.0)
parser.add_argument('--coef', type=float, default=1.0, help='coefficient of variants proportion')

# distill configs
parser.add_argument('--kd', action='store_true', help='use self-distillation for fine-tuning')
parser.add_argument('--b', type=float, default=0.5)
parser.add_argument('--T', type=float, default=1.0)

parser.add_argument('--arch_t', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--path_t', default=None, type=str, help='path to teacher model')
parser.add_argument('--feat_t', default=None, type=int,
                    help='last feature dim of teacher')

# moco specific configs:
parser.add_argument('--moco', dest='moco', action='store_true', help='build moco model')
parser.add_argument('--moco-dim', default=32, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=4096, type=int,
                    help='queue size; number of negative keys (default: 1024)')
parser.add_argument('--base_k', default=2, type=int,
                    help='minimum number of contrast for each class')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=str2bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=str2bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=False, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=True, type=str2bool,
                    help='use cosine classifier')

# options for paco
parser.add_argument('--mark', default='test', type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='warmup epochs')
parser.add_argument('--aug', default='cifar100', type=str,
                    help='aug strategy')
parser.add_argument('--num_classes', default=100, type=int,
                    help='num classes in dataset')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--feat_dim', default=256, type=int,
                    help='last feature dim of backbone')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='weight for kl_div loss (default: 1.0)')
parser.add_argument('--beta', default=1.0, type=float,
                    help='weight for contrast loss (default: 1.0)')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='weight for cross-entropy loss (default: 1.0)')


parser.add_argument('--epoch-multiplier', default=1, type=float,
                    help='multiply epoch by times')


class NormedLinearPruner(tp.pruner.BasePruningFunc):
    TARGET_MODULES = NormedLinear

    # def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
    #     # keep_idxs = list(set(range(layer.weight.data.shape[-1]) - set(idxs)))
    #     # layer.weight.data = layer.weight.data[:, keep_idxs]
    #     return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        # keep_idxs = list(set(range(layer.weight.data.shape[0])) - set(idxs))
        # layer.weight.data = layer.weight.data[keep_idxs, :]
        return layer

    def get_in_channels(self, layer):
        return layer.weight.data.shape[0]
        # return 64

    def get_out_channels(self, layer):
        return layer.weight.data.shape[-1]
        # return 100

    prune_out_channels = prune_in_channels


def init_env(args):
    args.epochs = int(args.epochs * args.epoch_multiplier)
    args.warmup_epochs *= int(args.epochs * args.epoch_multiplier)

    args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
    os.makedirs(args.root_model, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    def print_pass(*args):
        tqdm.write(' '.join([str(v) for v in args]), file=sys.stdout)

    builtins.print = print_pass
    args.is_master = True
    tb_logger = SummaryWriter(os.path.join(args.root_model, 'tb_logs'), flush_secs=2)

    return tb_logger


def set_dataset_moco(args):
    augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    CIFAR = datasets.CIFAR100 \
        if args.dataset == 'cifar100' else datasets.CIFAR10

    val_dataset = CIFAR(
        root=args.data,
        train=False,
        download=True,
        transform=val_transform)

    transform_train = [transforms.Compose(augmentation_regular),
                       transforms.Compose(augmentation_regular),
                       transforms.Compose(augmentation_regular)]

    ImbalanceCIFAR = ImbalanceCIFAR100_MOCO \
        if args.dataset == 'cifar100' else ImbalanceCIFAR10_MOCO
    train_dataset = ImbalanceCIFAR(root=args.data, imb_type='exp', imb_factor=args.imb_factor, rand_number=0,
                                   train=True, download=True, transform=transform_train)
    print(transform_train)

    print(f'===> Training data length {len(train_dataset)}')

    # criterion.cal_weight_for_classes(train_dataset.get_cls_num_list())
    # args.cls_weight = criterion.cls_weight

    return train_dataset, val_dataset


def set_model(args):
    print("=> creating model '{}'".format(args.arch))
    if 'cifar' in args.dataset:
        model = getattr(resnet_cifar, args.arch)(num_classes=args.num_classes, use_norm=True)
    else:
        model = getattr(resnet_imagenet, args.arch)(num_classes=args.num_classes, use_norm=True)
    # model = resnet.resnet18(pretrained=True)

    model = model.to(args.device)

    print("Use {} for training".format(args.device))

    if args.resume is not None:
        checkpoint = torch.load(f'{args.root_model}/{args.resume}', map_location=args.device)
        tp.load_state_dict(model, state_dict=checkpoint['state_dict'])
        print('=> resume checkpoint from {}'.format(args.resume))

    else:
        assert args.pretrained is not None and os.path.isfile(args.pretrained), 'pretrained model is necessary'
        checkpoint = torch.load(args.pretrained, map_location=args.device)
        if 'moco' in args.pretrained:
            ckpt = {}
            for key in checkpoint['state_dict']:
                if 'encoder_q' in key:
                    ckpt[key.replace('encoder_q.', '')] = checkpoint['state_dict'][key]
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.pretrained, checkpoint['epoch']))

    cudnn.benchmark = True

    print(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    return model, optimizer


def set_moco_model(args):
    print("=> creating model '{}'".format(args.arch))
    if 'cifar' in args.dataset:
        model = moco.builder.MoCo(
            getattr(resnet_cifar, args.arch),
            args.moco_dim, args.moco_k, args.moco_m, args.mlp,
            args.feat_dim, args.feat_t, args.normalize, args.num_classes)
    else:
        # model = getattr(resnet_imagenet, args.arch)(num_classes=args.num_classes, use_norm=True)
        model = moco.builder.MoCo(
            getattr(resnet_imagenet, args.arch),
            args.moco_dim, args.moco_k, args.moco_m, args.mlp,
            args.feat_dim, args.feat_t, args.normalize, args.num_classes)
    # model = resnet.resnet18(pretrained=True)

    print(model)
    model = model.to(args.device)

    print("Use {} for training".format(args.device))

    if args.resume is not None:
        checkpoint = torch.load(f'{args.root_model}/{args.resume}', map_location=args.device)
        tp.load_state_dict(model, state_dict=checkpoint['state_dict'])
        print('=> resume checkpoint from {}'.format(args.resume))

    else:
        assert args.pretrained is not None and os.path.isfile(args.pretrained), 'pretrained model is necessary'
        checkpoint = torch.load(args.pretrained, map_location=args.device)

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        model.encoder_k = model.encoder_q

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.pretrained, checkpoint['epoch']))

    cudnn.benchmark = True

    # print(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    return model, optimizer


def set_pruner(args, model):
    if 'cifar' in args.dataset:
        example_inputs = torch.randn(1, 3, 32, 32).to(args.device)
    else:
        example_inputs = torch.randn(1, 3, 256, 256).to(args.device)

    if args.imp == 'taylor' or args.imp == 'taylor_v2' or args.imp == 'merge' or args.imp == 'cic' or args.imp == 'cls-aware':
        imp = tp.importance.GroupTaylorImportance(multivariable=False)
    elif args.imp == 'l1':
        imp = tp.importance.GroupNormImportance(p=1)
    elif args.imp == 'l2':
        imp = tp.importance.GroupNormImportance()
    elif args.imp == 'hessian':
        imp = tp.importance.HessianImportance()
    elif args.imp == 'bns':
        imp = tp.importance.BNScaleImportance()
    elif args.imp == 'fpgm':
        imp = FPGMImportance()

    else:
        assert 0, 'unsupported importance criterion'

    if args.moco:
        model = model.encoder_q

    ignored_layers = []
    if args.stage == 1:
        for name, module in model.named_modules():
            if 'layer4' in name:
                ignored_layers.append(module)

    elif args.stage == 2:
        for name, module in model.named_modules():
            if 'layer1' in name or 'layer3' in name or 'layer2' in name:
                ignored_layers.append(module)
    else:
        for name, module in model.named_modules():
            if 'layer4' in name and 'conv2' in name:
                ignored_layers.append(module)
    ignored_layers.append(model.fc)

    customed_pruner = NormedLinearPruner()

    print('ignored layers: ', ignored_layers)

    if args.imp == 'taylor_v2':
        pruner = TaylorStepPruner(model, example_inputs,
                                  iterative_steps=args.iterative_steps,
                                  global_pruning=args.global_pruning,
                                  importance=imp,
                                  customized_pruners={NormedLinear: customed_pruner},
                                  # unwrapped_parameters=unwrapped_parameters,
                                  pruning_ratio=args.pruning_ratio,
                                  ignored_layers=ignored_layers)
    elif args.imp == 'fpgm':
        pruner = FPGMPruner(model, example_inputs,
                            global_pruning=args.global_pruning,
                            importance=imp,
                            customized_pruners={NormedLinear: customed_pruner},
                            # unwrapped_parameters=unwrapped_parameters,
                            pruning_ratio=args.pruning_ratio * 1.6,
                            similar_pruning_ratio=(args.pruning_ratio * 1.6 / 3),
                            ignored_layers=ignored_layers)
    elif args.imp == 'merge':
        pruner = TaylorMergePruner(model, example_inputs,
                                  iterative_steps=args.iterative_steps,
                                  global_pruning=args.global_pruning,
                                  importance=imp,
                                  customized_pruners={NormedLinear: customed_pruner},
                                  # unwrapped_parameters=unwrapped_parameters,
                                  pruning_ratio=args.pruning_ratio,
                                  ignored_layers=ignored_layers)
    elif args.imp == 'cic':
        pruner = CICPruner(model, example_inputs,
                                  iterative_steps=args.iterative_steps,
                                  global_pruning=args.global_pruning,
                                  importance=imp,
                                  customized_pruners={NormedLinear: customed_pruner},
                                  # unwrapped_parameters=unwrapped_parameters,
                                  pruning_ratio=args.pruning_ratio,
                                  ignored_layers=ignored_layers)
        pruner.set_balance_alpha(args.coef)
    elif args.imp == 'cls-aware':
        pruner = ClassAwarePruner(model, example_inputs,
                                  iterative_steps=args.iterative_steps,
                                  global_pruning=args.global_pruning,
                                  importance=imp,
                                  customized_pruners={NormedLinear: customed_pruner},
                                  # unwrapped_parameters=unwrapped_parameters,
                                  pruning_ratio=args.pruning_ratio,
                                  ignored_layers=ignored_layers)
    else:
        pruner = tp.pruner.MetaPruner(model, example_inputs,
                                      iterative_steps=args.iterative_steps,
                                      global_pruning=args.global_pruning,
                                      importance=imp,
                                      customized_pruners={NormedLinear: customed_pruner},
                                      # unwrapped_parameters=unwrapped_parameters,
                                      pruning_ratio=args.pruning_ratio,
                                      ignored_layers=ignored_layers)

    return pruner, example_inputs


def fine_tuning(model, train_loader, val_loader, optimizer, criterion, tb_logger, args, teacher=None):
    best_acc1 = 0
    criterion_ce = nn.CrossEntropyLoss().to(args.device)
    # args.epochs = int(args.epochs * args.epoch_multiplier)
    for epoch in tqdm(range(args.start_epoch, args.epochs), position=0, leave=True, disable=not args.is_master):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        if args.moco:
            moco_train(train_loader, model, criterion, optimizer, tb_logger, epoch, args)
        else:
            if args.kd:
                assert teacher is not None, 'teacher should not be None if arg.kd is true'
                train(train_loader, model, criterion, optimizer, tb_logger, epoch, args, teacher=teacher)
            else:
                train(train_loader, model, criterion, optimizer, tb_logger, epoch, args)

        if (epoch + 1) % args.val_freq == 0:
            acc1 = validate(val_loader, train_loader, model, criterion_ce, tb_logger, epoch, args)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_cur = 'Current Prec@1: %.3f\n' % (acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_cur, output_best)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': tp.state_dict(model),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=f'{args.root_model}/finetune_ckpt_s{args.stage}.pth')

    return


def train(train_loader, model, criterion, optimizer, tb_logger, epoch, args, pruner=None, iter=None, teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    if pruner is not None:
        imp = pruner.importance
        if isinstance(imp, tp.importance.HessianImportance):
            imp.zero_grad()
            criterion.reduction = 'none'
        if isinstance(pruner, TaylorStepPruner):
            pruner.set_storage()
        if isinstance(pruner, TaylorMergePruner):
            pruner.set_storage(args.num_type)
    if args.kd:
        distill_kl = DistillKL(args.T)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader, position=1, leave=False, disable=not args.is_master)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.device is not None:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

        # compute output
        logits = model(images)
        if pruner is not None:
            loss = criterion(logits, target)
        else:
            loss = criterion(logits * 30 + args.cls_weight.log(), target)
        if args.kd:
            t_logits = teacher(images)
            kd_loss = distill_kl(logits, t_logits)
            loss = loss + args.b * kd_loss

        if pruner is not None:
            model.zero_grad()
            loss.backward()
            if args.imp == 'taylor_v2':
                pruner.store_importance()
            elif args.imp == 'merge':
                pruner.store_importance(args.num_type)
                # print(target)
            elif args.imp == 'cic' or args.imp == 'cls-aware':
                pruner.store_importance(target[0])
            else:
                pass
        else:
            # compute gradient and do SGD step
            if optimizer is not None:
                optimizer.zero_grad()
            loss.backward()

        if optimizer is not None:
            optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter is not None and i + 1 >= iter:
            break

        if i % args.print_freq == 0:
            progress.display(i, args)
            global_step = epoch * len(train_loader) + i
            if tb_logger is not None:
                tb_logger.add_scalar('Train/losses', losses.avg, global_step)
                tb_logger.add_scalar('Train/top1', top1.avg, global_step)
                tb_logger.add_scalar('Train/top5', top5.avg, global_step)



def moco_train(train_loader, model, criterion, optimizer, tb_logger, epoch, args, pruner=None, iter=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # print(len(train_loader))

    # switch to train mode
    model.train()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    if pruner is not None:
        imp = pruner.importance
        # model.eval()
        if isinstance(imp, tp.importance.HessianImportance):
            imp.zero_grad()
            criterion.reduction = 'none'
        if isinstance(pruner, TaylorStepPruner):
            pruner.set_storage()
        if isinstance(pruner, TaylorMergePruner):
            pruner.set_storage(args.num_type)
    end = time.time()
    for i, (images, target, in_idx) in enumerate(tqdm(train_loader, position=1, leave=False, disable=not args.is_master)):
        # measure data loading time
        data_time.update(time.time() - end)

        # print(len(images))

        if args.device is not None:
            images = [img.cuda(args.device, non_blocking=True) for img in images]
            target = target.cuda(args.device, non_blocking=True)
            in_idx = in_idx.cuda(args.device, non_blocking=True)

        # compute output
        query, key, k_labels, k_idx, logits, t_logits = model(*images, target, in_idx)
        loss = criterion(query, target, in_idx, key, k_labels, k_idx, logits, t_logits)

        total_logits = torch.cat((total_logits, logits))
        total_labels = torch.cat((total_labels, target))

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # compute gradient and do SGD step
        if pruner is not None:
            model.zero_grad()
            loss.backward()
            if args.imp == 'taylor_v2':
                pruner.store_importance()
            elif args.imp == 'merge':
                pruner.store_importance(args.num_type)
            elif args.imp == 'cic' or args.imp == 'cls-aware':
                pruner.store_importance(target[0])
            else:
                pass
        else:
            if optimizer is not None:
                optimizer.zero_grad()
            loss.backward()
        if optimizer is not None:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter is not None and i + 1 >= iter:
            break

        if i % args.print_freq == 0:
            progress.display(i, args)
            global_step = epoch * len(train_loader) + i
            if tb_logger is not None:
                tb_logger.add_scalar('Train/losses', losses.avg, global_step)
                tb_logger.add_scalar('Train/top1', top1.avg, global_step)
                tb_logger.add_scalar('Train/top5', top5.avg, global_step)


def validate(val_loader, train_loader, model, criterion, tb_logger, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader, position=1, leave=False, disable=not args.is_master)):
            if args.device is not None:
                images = images.to(args.device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.to(args.device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        open(args.root_model + "/" + args.mark + ".log", "a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                                                                     .format(top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'.format(top1=top1, top5=top5))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(preds, total_labels, train_loader,
                                                                          acc_per_cls=True)

        print('many acc:{:.3f} median acc:{:.3f} low acc:{:.3f}'.format(many_acc_top1, median_acc_top1, low_acc_top1))

        if args.evaluate:
            with open(args.eval_log, 'a+') as log_file:
                log_file.write('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} \n'.format(top1=top1, top5=top5))
                log_file.write('many acc:{:.3f} median acc:{:.3f} low acc:{:.3f}\n'.format(many_acc_top1, median_acc_top1, low_acc_top1))
                log_file.write('*'*100 + '\n')

        if tb_logger is not None:
            tb_logger.add_scalar('Validation/Acc@1', top1.avg, epoch)
            tb_logger.add_scalar('Validation/Acc@5', top5.avg, epoch)
            tb_logger.add_scalar('Validation/Many_acc', many_acc_top1, epoch)
            tb_logger.add_scalar('Validation/Medium_acc', median_acc_top1, epoch)
            tb_logger.add_scalar('Validation/Low_acc', low_acc_top1, epoch)

    return top1.avg


def set_dataset_type(train_dataset, many, mid, few, args):

    dataset = copy.deepcopy(train_dataset)

    print('full length:', len(dataset))
    if args.num_type == 'all':
        return dataset

    if args.num_type == 'many':
        dataset.data = [train_dataset.data[img] for img in range(many)]
        dataset.targets = [train_dataset.targets[tgt] for tgt in range(many)]
    elif args.num_type == 'mid':
        dataset.data = [train_dataset.data[img] for img in range(many, mid)]
        dataset.targets = [train_dataset.targets[tgt] for tgt in range(many, mid)]
    elif args.num_type == 'few':
        dataset.data = [train_dataset.data[img] for img in range(mid, few)]
        dataset.targets = [train_dataset.targets[tgt] for tgt in range(mid, few)]

    print('new length:', len(dataset))

    return dataset


def get_many_mid_few_num(dataset):
    print(dataset.cls_num_list)
    cnl = torch.tensor(dataset.cls_num_list)
    many_thr_idx = (cnl > 100).sum()
    few_thr_idx = (cnl >= 20).sum()
    many, mid, few = cnl[:many_thr_idx].sum(), cnl[many_thr_idx:few_thr_idx].sum(), cnl[few_thr_idx:].sum()
    print('num many: {}, num_mid: {}, num few: {}'.format(many, mid, few))
    # assert 0
    return many, many + mid, many + mid + few


def main():
    # init config and environment
    args = parser.parse_args()
    tb_logger = init_env(args)
    # set model
    if args.moco:
        model, optimizer = set_moco_model(args)
    else:
        model, optimizer = set_model(args)

    teacher_model = copy.deepcopy(model) if args.kd else None
    if args.kd:
        teacher_model.eval()
        print('kd: {} b: {} T:{}'.format(args.kd, args.b, args.T))

    # set dataloader
    if 'cifar' in args.dataset:
        train_dataset, val_dataset = set_dataset_moco(args) if args.moco else set_dataset_cifar(args)
    else:
        train_dataset, val_dataset = set_dataset_imagenet(args)

    criterion_ce = nn.CrossEntropyLoss().to(args.device)
    if args.moco:
        criterion = GMLLoss(args.beta, args.gamma, args.moco_t, args.num_classes, args.alpha).cuda(args.device)
        criterion.cal_weight_for_classes(train_dataset.cls_num_list)
        if args.base_k > 0:
            model.set_cls_weight(criterion.cls_weight, args.base_k)
        args.cls_weight = criterion.cls_weight
    else:
        criterion = criterion_ce
        gml = GMLLoss(0, 0, args.moco_t, args.num_classes, 0).to(args.device)
        gml.cal_weight_for_classes(train_dataset.cls_num_list)
        args.cls_weight = gml.cls_weight

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # set pruner
    pruner, example_inputs = set_pruner(args, model)
    macs, params = tp.utils.count_ops_and_params(pruner.model, example_inputs)
    print("MACS: {} G, Params: {} M \n".format(macs / 1e9, params / 1e6))

    if args.evaluate:
        with open(args.eval_log, 'a+') as log_file:
            log_file.write('*'*100 + '\n')
            log_file.write(args.mark + '\n')
            log_file.write("MACS: {} G, Params: {} M \n".format(macs / 1e9, params / 1e6))

        print(" start evaluation **** ")
        validate(val_loader, train_loader, model, criterion_ce, tb_logger, 0, args)
        return

    # do pruning
    if args.resume is None or args.stage != 1:
        for i in range(args.iterative_steps):
            if args.imp == 'taylor' or args.imp == 'taylor_v2' or args.imp == 'hessian':
                start = time.time()
                if 'cifar' in args.dataset:
                    many, mid, few = get_many_mid_few_num(train_dataset)
                    new_train_dataset = set_dataset_type(train_dataset, many, mid, few, args)
                    new_dataloader = torch.utils.data.DataLoader(
                                        new_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True)
                    if args.moco:
                        moco_train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                    else:
                        train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                else:
                    train_dataset.set_num_type(args.num_type)
                    # train_loader.dataset.set_num_type(args.num_type)
                    new_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=False,
                        persistent_workers=True)
                    print('{} num data: {}'.format(args.num_type, len(train_dataset)))
                    if args.moco:
                        moco_train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                    else:
                        train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)

                    train_dataset.set_num_type('all')
                    # print('All num data: {}'.format(len(train_loader)))
                end = time.time()
                print('time ocst:', end - start)

            if args.imp == 'fpgm':
                pruner.step()
                macs, params = tp.utils.count_ops_and_params(pruner.model, example_inputs)
                print("MACS: {} G, Params: {} M".format(macs / 1e9, params / 1e6))
                acc1 = validate(val_loader, train_loader, model, criterion, tb_logger, 0, args)
                break

            if args.imp == 'merge':
                if 'cifar' in args.dataset:
                    many, mid, few = get_many_mid_few_num(train_dataset)
                    for num_type in ['many', 'mid', 'few']:
                        args.num_type = num_type
                        new_train_dataset = set_dataset_type(train_dataset, many, mid, few, args)
                        new_dataloader = torch.utils.data.DataLoader(
                            new_train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=False,
                            persistent_workers=True)
                        if args.moco:
                            moco_train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                        else:
                            train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                        del new_train_dataset
                        del new_dataloader
                else:
                    for num_type in ['many', 'mid', 'few']:
                        train_dataset.set_num_type(num_type)
                        args.num_type = num_type
                        # train_loader.dataset.set_num_type(args.num_type)
                        new_dataloader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=False, drop_last=False,
                            persistent_workers=False)
                        # train_loader.dataset.set_num_type(num_type)
                        if args.moco:
                            moco_train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                        else:
                            train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)

                for ii, (g_many, g_mid, g_few) in enumerate(zip(pruner.step(interactive=True, num_type='many'),
                                                  pruner.step(interactive=True, num_type='mid'),
                                                  pruner.step(interactive=True, num_type='few'))):
                    for jj in range(len(g_many._group)):
                        dep, idx_many = g_many._group[jj]
                        _, idx_mid = g_mid._group[jj]
                        _, idx_few = g_few._group[jj]
                        idx = list(set(idx_many).intersection(set(idx_mid), set(idx_few)))
                        # idx = list(set(idx_many).intersection(set(idx_few)))
                        rate = len(idx) / len(idx_few)
                        print(ii, jj, 'rate: ', rate)
                        if rate > 0.0:
                            g_many._group[jj] = tp._helpers.GroupItem(dep, idx)
                        else:
                            g_many._group[jj] = tp._helpers.GroupItem(dep, idx_few)
                    g_many.prune()

            if args.imp == 'cic' or args.imp == 'cls-aware':
                num_cls = len(args.cls_weight)
                pruner.set_clswise_storage(num_cls, args.cls_weight)

                start = time.time()
                # if 'cifar' in args.dataset:
                if False:
                    for cls in range(num_cls):
                        train_dataset.set_cls_dataset(cls)
                        args.num_type = cls
                        new_dataloader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=False, drop_last=False,
                            persistent_workers=False)
                        if args.moco:
                            moco_train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                        else:
                            train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                    end = time.time()
                    print('time cost: {}'.format(end - start))
                else:
                    train_dataset.reorganize_cls_sqe(args.cic_batch)
                    new_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.cic_batch, shuffle=False,
                        num_workers=16, pin_memory=False, drop_last=False,
                        persistent_workers=False)
                    if args.moco:
                        moco_train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                    else:
                        train(new_dataloader, model, criterion, None, tb_logger, 0, args, pruner=pruner)
                    end = time.time()
                    print('time cost: {}'.format(end - start))

                train_dataset.reset_data()

            if args.imp != 'merge':
                for g in pruner.step(interactive=True):
                    g.prune()

        macs, params = tp.utils.count_ops_and_params(pruner.model, example_inputs)
        print("MACS: {} G, Params: {} M".format(macs / 1e9, params / 1e6))

    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': tp.state_dict(model),
        'optimizer': None,
    }, is_best=False, filename=f'{args.root_model}/pruned.pth')

    print(model)

    # acc1 = validate(val_loader, train_loader, model, criterion, tb_logger, 0, args)
    if args.moco:
        criterion.alpha, criterion.beta, criterion.gamma = 1, 1, 1

    # finetune
    fine_tuning(model, train_loader, val_loader, optimizer, criterion, tb_logger, args, teacher=teacher_model)

    return


if __name__ == '__main__':
    main()
