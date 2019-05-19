import argparse
import os
import random
import time

# import gluoncvth as gcv
import numpy as np
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
from cleverhans.attacks import MadryEtAl, ProjectedGradientDescent
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PGD Attack')
parser.add_argument('--gpu', '--gpu_ids', type=str, required=True)
parser.add_argument('--imagenet_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('-p', '--phase', default='train', type=str)
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('-m', '--model', type=str, choices=['AlexNet', 'VGG16', 'ResNet50', 'DenseNet121', 'Inception_v3'],
                    required=True)

args = parser.parse_args()

IMAGENET_DIR = args.imagenet_dir
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAVE_DIR = args.save_dir
PHASE = args.phase
BATCH_SIZE = args.batch_size
"""note the GPU type problem!!!"""

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

model_list = {
    'AlexNet': models.alexnet,
    'VGG16': models.vgg16_bn,
    'ResNet50': models.resnet50,
    'DenseNet121': models.densenet121,
    'Inception_v3': models.inception_v3
}
model_weights = {
    'AlexNet': 'xxx/alexnet-owt-4df8aa71.pth',
    'VGG16': 'xxx/vgg16_bn-6c64b313.pth',
    'ResNet50': 'xxx/resnet50-19c8e357.pth',
    'DenseNet121': 'xxx/densenet121-a639ec97.pth',
    'Inception_v3': 'xxx/inception_v3_google-1a9a5a14.pth'
}


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

if 'inc' in args.model.lower():
    transform = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor()
    ])
else:
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
model = model_list[args.model]()
model.load_state_dict(torch.load(model_weights[args.model]))
# model = gcv.models.resnet50(pretrained=True, dilated=False, deep_base=False)
clf = Classifier(model)
clf.to(device)
clf = nn.DataParallel(clf)
clf.eval()

print('configuring TensorFlow...')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

x_op = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
y_op = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
onehot_op = tf.one_hot(y_op, 1000)

tf_model_fn = convert_pytorch_model_to_tf(clf)
cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

# pgd_op = MadryEtAl(cleverhans_model, sess=sess)
pgd_op = ProjectedGradientDescent(cleverhans_model, sess=sess, default_rand_init=True)
pgd_params = {'eps': 16 / 255.0,
              'eps_iter': 2 / 255.0,
              'nb_iter': 10,
              'clip_min': 0.0,
              'clip_max': 1.0}
adv_x_op = pgd_op.generate(x_op, y=onehot_op, **pgd_params)

clean_logits_op = tf_model_fn(x_op)
adv_logits_op = tf_model_fn(adv_x_op)

cudnn.benchmark = True

total = -1
for step, (images, labels) in enumerate(train_loader):
    print('To be attacked: {}th, {}'.format(total + 1, os.path.basename(train_set.imgs[total + 1][0])))
    start = time.time()

    images = images.to(device)
    labels = labels.to(device)

    # ==================================================================================================================
    # adv_x = sess.run(adv_x_op, feed_dict={x_op: images, y_op: labels})
    # adv_x = torch.from_numpy(adv_x)
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
    try:
        adv_x = sess.run(adv_x_op, feed_dict={x_op: images, y_op: labels})
    except:
        y_op = tf.placeholder(tf.int64, shape=(images.size(0),))
        onehot_op = tf.one_hot(y_op, 1000)
        adv_x_op = pgd_op.generate(x_op, y=onehot_op, **pgd_params)
        adv_x = sess.run(adv_x_op, feed_dict={x_op: images, y_op: labels})
    adv_x = torch.from_numpy(adv_x).to(device)

    with torch.no_grad():
        clean_logits = clf(images)
        adv_logits = clf(adv_x)

    for n in range(images.size(0)):
        total += 1
        filename = os.path.basename(train_set.imgs[total][0]).replace('.JPEG', '.npy')
        classname = train_set.classes[labels[n].item()]

        directory = os.path.join(SAVE_DIR, 'clean/', PHASE, classname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, filename), clean_logits[n].cpu().numpy())

        directory = os.path.join(SAVE_DIR, 'adv/', PHASE, classname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, filename), adv_logits[n].cpu().numpy())

    print('batch time: {:.3f}s'.format(time.time() - start))
