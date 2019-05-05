import argparse
import csv
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mixed_folder import PairedDatasetFolder
from network import MLP
from utils import Visualizer

parser = argparse.ArgumentParser(description='Adversarial MLP Training')
parser.add_argument('--gpu', '--gpu_ids', type=str, required=True)
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('-n', '--n_layers', default=2, type=int)
parser.add_argument('-w', '--width', default=2048, type=int)
parser.add_argument('--lam', '--lambda', default=0.5, type=float)
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('-f', '--print_freq', default=50, type=int)
parser.add_argument('--name', default='mlp', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--checkpoint', type=str)

args = parser.parse_args()
print('\nArguments:')
print(args)

IMAGENET_ADV = '/DATA5_DB8/data/yfli/datasets/ImageNet_PGD16/VGG16/adv/'
IMAGENET_CLEAN = '/DATA5_DB8/data/yfli/datasets/ImageNet_PGD16/VGG16/clean/'
IMAGENET_VAL = '/DATA5_DB8/data/yfli/datasets/ImageNet_val_PGD/VGG16/'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    directory = os.path.join('checkpoints/', args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(directory, 'best.tar'))


vis = Visualizer(env=args.name, server='http://202.120.39.167', port=8099)

print('\nsetting seeds...')
vis.log('setting seeds...')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

"""note that val samples are already in 224!!!"""
transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor()
])
print('loading datasets...')
vis.log('loading datasets...')
if True:  # for debugging
    train_set = PairedDatasetFolder(os.path.join(IMAGENET_ADV, 'train/'), os.path.join(IMAGENET_CLEAN, 'train/'),
                                    loader=np.load, extensions=['.npy'], transform=torch.from_numpy)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val_set = datasets.ImageFolder(os.path.join(IMAGENET_VAL, 'val/'), transform=transform)
val_set = datasets.DatasetFolder(os.path.join(IMAGENET_VAL, 'val/'), loader=np.load, extensions=['.npy'],
                                 transform=torch.from_numpy)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('building models...')
vis.log('building models...')
clf = models.resnet50(pretrained=True)
# clf.avgpool = nn.AdaptiveAvgPool2d(1)
for param in clf.parameters():
    param.requires_grad = False
clf.to(device)
clf.eval()
clf_norm = Normalization(IMAGENET_MEAN, IMAGENET_STD)
clf_norm.to(device)
clf_norm.eval()
"""fixed"""

net = MLP(n_layers=args.n_layers, width=args.width)
net.to(device)
net = nn.DataParallel(net)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.wd)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ckpt = torch.load(args.checkpoint)
# net.load_state_dict(ckpt['model'])
# optimizer.load_state_dict(ckpt['optimizer'])
# train_loss = ckpt['train_loss']
# test_loss = ckpt['test_loss']
# test_acc = ckpt['test_acc']
# epochs = ckpt['epoch']

if not os.path.exists('logs/'):
    os.makedirs('logs/')
log_name = 'logs/' + args.name + '_' + str(args.seed) + '.csv'
if not os.path.exists(log_name):
    with open(log_name, 'w') as log_file:
        log_writer = csv.writer(log_file, delimiter=',')
        log_writer.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc', 'best'])

train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0.0

cudnn.benchmark = True

for epoch in range(args.epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, args.epochs))
    print('-' * 20)
    vis.log('\nEpoch {}/{}'.format(epoch + 1, args.epochs))
    vis.log('-' * 20)

    # scheduler.step()

    # training phase
    print('training phase')
    vis.log('training phase')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    net.train()

    end = time.time()
    for step, (logits_adv, logits_clean, labels) in enumerate(train_loader):
        logits_adv = logits_adv.to(device)
        logits_clean = logits_clean.to(device)
        labels = labels.to(device)

        # logits = logits - logits.max(dim=1, keepdim=True)[0]  # Normalizing by reducing the maximum
        # logits = logits / logits.norm(p=2, dim=1, keepdim=True)  # L2 normalization

        output_adv = net(logits_adv)
        output_clean = net(logits_clean)
        loss = args.lam * criterion(output_clean, labels) + (1 - args.lam) * criterion(output_adv, labels)

        _, pred_adv = output_adv.max(dim=1)
        correct_adv = pred_adv.eq(labels)
        _, pred_clean = output_clean.max(dim=1)
        correct_clean = pred_clean.eq(labels)
        acc = args.lam * correct_clean.double().mean() * 100.0 + (1 - args.lam) * correct_adv.double().mean() * 100.0
        losses.update(loss.item(), logits_adv.size(0))
        top1.update(acc.item(), logits_adv.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'.format(
                epoch, step, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
            vis.log('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'.format(
                epoch, step, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))

    train_loss = losses.avg
    train_acc = top1.avg
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    vis.plot_many({'train loss': train_loss, 'train acc': train_acc})

    # validation phase
    print('validation phase')
    vis.log('validation phase')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    net.eval()

    end = time.time()
    for step, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = images
            # logits = clf(clf_norm(images))
            # logits = logits - logits.max(dim=1, keepdim=True)[0]  # Normalizing by reducing the maximum
            # logits = logits / logits.norm(p=2, dim=1, keepdim=True)  # L2 normalization
            output = net(logits)
            loss = criterion(output, labels)

        _, pred = output.max(dim=1)
        correct = pred.eq(labels)
        acc = correct.double().mean() * 100.0
        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'.format(
                step, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
            vis.log('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'.format(
                step, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    test_loss = losses.avg
    test_acc = top1.avg
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(' * Acc@1 {top1.avg:.2f}%'.format(top1=top1))
    vis.log(' * Acc@1 {top1.avg:.2f}%'.format(top1=top1))
    vis.plot_many({'test loss': test_loss, 'test acc': test_acc})

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    state = {'args': args, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
             'train_losses': train_losses, 'train_accs': train_accs, 'test_losses': test_losses, 'test_accs': test_accs,
             'best_acc': best_acc}
    save_checkpoint(state, is_best)

    with open(log_name, 'a') as log_file:
        log_writer = csv.writer(log_file, delimiter=',')
        log_writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc, best_acc])

print('\nBest accuracy: {:.2f}%'.format(best_acc))
vis.log('\nBest accuracy: {:.2f}%'.format(best_acc))
