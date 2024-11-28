import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from randaugment import rand_augment_transform
import torchvision.models as models
import torch.nn.functional as F
from models import resnet_cifar, resnet_imagenet
from autoaug import CIFAR10Policy, Cutout
import moco.loader
import moco.builder
from dataset.imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
from dataset.imagenet import ImageNetLT
from dataset.imagenet_moco import ImageNetLT_moco
from dataset.inat import INaturalist
from dataset.inat_moco import INaturalist_moco
import torchvision.datasets as datasets
from utils import shot_acc
from losses import GMLLoss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model_names += ['resnext101_32x4d']
model_names += ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', choices=['inat', 'imagenet', 'cifar100', 'cifar10'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./outputs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--val-freq', default=5, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--imb-factor', type=float,
                    metavar='IF', help='imbalanced factor', dest='imb_factor')
parser.add_argument('--lr', '--learning-rate', default=0.15, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', default=False, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--arch_t', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--path_t', default=None, type=str, help='path to teacher model')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=1024, type=int,
                    help='queue size; number of negative keys (default: 1024)')
parser.add_argument('--base_k', default=0, type=int,
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
parser.add_argument('--normalize', default=True, type=str2bool,
                    help='use cosine classifier')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='warmup epochs')
parser.add_argument('--aug', default='cifar100', type=str,
                    help='aug strategy')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='num classes in dataset')
parser.add_argument('--feat_dim', default=64, type=int,
                    help='last feature dim of backbone')

parser.add_argument('--epoch-multiplier', default=1, type=int,
                    help='multiply epoch by times')


def set_dataset_imagenet(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    txt_train = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_train.txt' if args.dataset == 'inat' \
        else f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_train.txt'

    txt_test = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_val.txt' if args.dataset == 'inat' \
        else f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_test.txt'

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if args.dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    val_dataset = INaturalist(
        root=args.data,
        txt=txt_test,
        transform=val_transform
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_test,
        transform=val_transform)


    if args.moco:
        if args.aug == 'randcls':
             transform_train = [
                 transforms.Compose(augmentation_randncls),
                 transforms.Compose(augmentation_randnclsstack),
                 transforms.Compose(augmentation_randnclsstack),
             ]
        elif args.aug == 'randclsstack':
             transform_train = [
                 transforms.Compose(augmentation_randnclsstack),
                 transforms.Compose(augmentation_randnclsstack),
                 transforms.Compose(augmentation_randnclsstack),
             ]
        elif args.aug == 'randcls_sim':
             transform_train = [
                 transforms.Compose(augmentation_randncls),
                 transforms.Compose(augmentation_sim),
                 transforms.Compose(augmentation_sim),
             ]
        else:
            raise NotImplementedError
    else:
        if args.aug == 'randcls':
            transform_train = transforms.Compose(augmentation_randncls)
        elif args.aug == 'randclsstack':
            transform_train = transforms.Compose(augmentation_randnclsstack)

        elif args.aug == 'randcls_sim':
            transform_train = transforms.Compose(augmentation_sim)
        else:
            raise NotImplementedError

    if not args.moco:
        train_dataset = INaturalist(
            root=args.data,
            txt=txt_train,
            transform=transform_train
        ) if args.dataset == 'inat' else ImageNetLT(
            root=args.data,
            txt=txt_train,
            transform=transform_train)
    else:
        train_dataset = INaturalist_moco(
            root=args.data,
            txt=txt_train,
            transform=transform_train
        ) if args.dataset == 'inat' else ImageNetLT_moco(
            root=args.data,
            txt=txt_train,
            transform=transform_train)
    print(f'===> Training data length {len(train_dataset)}')
    print(f'===> Validating data length {len(val_dataset)}')

    return train_dataset, val_dataset


def set_dataset_cifar(args):
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
        root='./cifar',
        train=False,
        download=True,
        transform=val_transform)

    transform_train = transforms.Compose(augmentation_regular)

    ImbalanceCIFAR = ImbalanceCIFAR100 \
        if args.dataset == 'cifar100' else ImbalanceCIFAR10
    train_dataset = ImbalanceCIFAR(root='./cifar', imb_type='exp', imb_factor=args.imb_factor, rand_number=0,
                                   train=True, download=True, transform=transform_train)
    print(transform_train)

    print(f'===> Training data length {len(train_dataset)}')

    return train_dataset, val_dataset


def main():
    args = parser.parse_args()
    args.epochs *= args.epoch_multiplier
    args.warmup_epochs *= args.epoch_multiplier

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

    main_worker(args.gpu, 1, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    def print_pass(*args):
        tqdm.write(' '.join([str(v) for v in args]), file=sys.stdout)

    builtins.print = print_pass
    args.is_master = True
    tb_logger = SummaryWriter(os.path.join(args.root_model, 'tb_logs'), flush_secs=2)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))

    print(args.num_classes)

    if 'cifar' in args.dataset:
        model = getattr(resnet_cifar, args.arch)(num_classes=args.num_classes, use_norm=True)  # origin: use_norm=True
    else:
        model = getattr(resnet_imagenet, args.arch)(num_classes=args.num_classes, use_norm=True) #

    print(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = GMLLoss(0, 0, args.moco_t, args.num_classes, 0).cuda(args.gpu)

    def get_wd_params(model: nn.Module):
        no_wd_params = list()
        wd_params = list()
        for n, p in model.named_parameters():
            if '__no_wd__' in n:
                no_wd_params.append(p)
            else:
                wd_params.append(p)

        return wd_params, no_wd_params

    wd_params, no_wd_params = get_wd_params(nn.ModuleList([model, criterion]))
    optimizer = torch.optim.SGD(wd_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if len(no_wd_params) > 0:
        optimizer_no_wd = torch.optim.SGD(no_wd_params, args.lr,
                                          momentum=args.momentum,
                                          weight_decay=0.0)
        [optimizer.add_param_group(pg) for pg in optimizer_no_wd.param_groups]

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if 'cifar' in args.dataset:
        train_dataset, val_dataset = set_dataset_cifar(args)
    else:
        train_dataset, val_dataset = set_dataset_imagenet(args)

    criterion.cal_weight_for_classes(train_dataset.cls_num_list)
    args.cls_weight = criterion.cls_weight
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    if args.evaluate:
        print(" start evaluation **** ")
        validate(val_loader, train_loader, model, criterion_ce, tb_logger, 0, args)
        return

    best_acc1 = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs), position=0, leave=True, disable=not args.is_master):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion_ce, optimizer, tb_logger, epoch, args)
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
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=f'{args.root_model}/ckpt.pth')

        # if (epoch + 1) % args.print_freq == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, is_best=False, filename=f'{args.root_model}/moco_ckpt_{(epoch+1):04d}.pth.tar')


def train(train_loader, model, criterion, optimizer, tb_logger, epoch, args):
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

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader, position=1, leave=False, disable=not args.is_master)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        # features, labels, logits, centers = model(*images, target)
        # loss = criterion(features, labels, logits, centers)
        logits = model(images)
        loss = criterion(logits * 30 + args.cls_weight.log(), target)
        # loss = criterion(logits, target)

        # total_logits = torch.cat((total_logits, logits))
        # total_labels = torch.cat((total_labels, target))

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

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
        print(' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f}\n'.format(top1=top1, top5=top5))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(preds, total_labels, train_loader,
                                                                          acc_per_cls=True)

        print('many acc:{:.4f} median acc:{:.4f} low acc:{:.4f}'.format(many_acc_top1, median_acc_top1, low_acc_top1))

        if tb_logger is not None:
            tb_logger.add_scalar('Validation/Acc@1', top1.avg, epoch)
            tb_logger.add_scalar('Validation/Acc@5', top5.avg, epoch)
            tb_logger.add_scalar('Validation/Many_acc', many_acc_top1, epoch)
            tb_logger.add_scalar('Validation/Medium_acc', median_acc_top1, epoch)
            tb_logger.add_scalar('Validation/Low_acc', low_acc_top1, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth', '_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(args.root_model + "/" + args.mark + ".log", "a+").write('\t'.join(entries) + "\n")
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if epoch <= args.warmup_epochs:
#         lr = args.lr / args.warmup_epochs * (epoch + 1)
#     elif epoch > args.schedule[1] * args.epoch_multiplier:
#         lr = args.lr * 0.01
#     elif epoch > args.schedule[0] * args.epoch_multiplier:
#         lr = args.lr * 0.1
#     else:
#         lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
       lr = lr / args.warmup_epochs * (epoch + 1 )
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1 ) / (args.epochs - args.warmup_epochs + 1 )))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
