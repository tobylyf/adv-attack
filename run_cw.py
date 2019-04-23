import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader

from carlini import CarliniWagnerL2

parser = argparse.ArgumentParser(description='C&W Attack')
parser.add_argument('--gpu', '--gpu_ids', type=str, required=True)
parser.add_argument('--imagenet_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('-p', '--phase', default='train', type=str)
parser.add_argument('-b', '--batch_size', default=256, type=int)

args = parser.parse_args()

IMAGENET_DIR = args.imagenet_dir
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAVE_DIR = args.save_dir
PHASE = args.phase
BATCH_SIZE = args.batch_size

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Classifier(nn.Module):
    def __init__(self, clf):
        super(Classifier, self).__init__()
        self.norm = Normalization(IMAGENET_MEAN, IMAGENET_STD)
        self.clf = clf
        for param in self.clf.parameters():
            param.requires_grad = False
        self.clf.eval()

    def forward(self, x):
        x = self.norm(x)
        x = self.clf(x)
        return x


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
print('loading dataset...')
train_set = datasets.ImageFolder(os.path.join(IMAGENET_DIR, PHASE), transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('building model...')
model = models.resnet50(pretrained=True)
clf = Classifier(model)
clf.to(device)
clf = nn.DataParallel(clf)
clf.eval()

attacker = CarliniWagnerL2((0.0, 1.0), 1000, learning_rate=0.01, search_steps=4, max_iterations=25,
                           initial_const=10, quantize=False, device=device)

cudnn.benchmark = True

total = -1
for step, (images, labels) in enumerate(train_loader):
    # if total < 4400:
    #     total += images.size(0)
    #     continue
    print('To be attacked: {}th, {}'.format(total + 1, os.path.basename(train_set.imgs[total + 1][0])))
    start = time.time()

    images = images.to(device)
    labels = labels.to(device)

    # ==================================================================================================================
    # adv_x = attacker.attack(clf, images, labels, targeted=False)
    #
    # for n in range(images.size(0)):
    #     total += 1
    #     filename = os.path.basename(train_set.imgs[total][0]).replace('.JPEG', '.png')
    #     classname = train_set.classes[labels[n].item()]
    #     directory = os.path.join(SAVE_DIR, classname)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     utils.save_image(adv_x[n], os.path.join(directory, filename))

    # ==================================================================================================================
    adv_x = attacker.attack(clf, images, labels, targeted=False)

    with torch.no_grad():
        clean_logits = clf(images)
        adv_logits = clf(adv_x)

    for n in range(images.size(0)):
        total += 1
        filename = os.path.basename(train_set.imgs[total][0]).replace('.JPEG', '.npy')
        classname = train_set.classes[labels[n].item()]

        directory = os.path.join(SAVE_DIR, 'image/', PHASE, classname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, filename), adv_x[n].cpu().numpy())

        directory = os.path.join(SAVE_DIR, 'clean/', PHASE, classname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, filename), clean_logits[n].cpu().numpy())

        directory = os.path.join(SAVE_DIR, 'adv/', PHASE, classname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, filename), adv_logits[n].cpu().numpy())

    print('batch time: {:.3f}s'.format(time.time() - start))
