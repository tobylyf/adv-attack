import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_layers=2, width=2048):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.width = width
        # TODO: Use Leaky ReLU? What Initialization?

        # first layer
        model = [nn.Linear(1000, width),
                 nn.ReLU(True),
                 nn.Dropout(0.5)]
        # middle layers
        if n_layers > 2:
            for i in range(n_layers - 2):
                model += [nn.Linear(width, width),
                          nn.ReLU(True),
                          nn.Dropout(0.5)]
        # last layer
        model += [nn.Linear(width, 1000)]

        self.classifier = nn.Sequential(*model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.classifier(x)


class MidFeature(nn.Module):
    def __init__(self):
        super(MidFeature, self).__init__()
        self.features = nn.Linear(2048, 1000)
        self.logits = nn.Linear(1000, 1000)

    def forward(self, features, logits):
        features = features.view(features.size(0), -1)
        features = self.features(features)
        logits = self.logits(logits)
        return features + logits


class Randomization(nn.Module):
    def __init__(self, p=0.5, size=330):
        super(Randomization, self).__init__()
        self.p = p
        self.size = size

    def forward(self, x):
        if torch.rand(1) < self.p:
            rnd = torch.randint(x.size(3), self.size, (1,), dtype=torch.int).item()
            x = F.interpolate(x, size=rnd, mode='nearest')
            pad_w = self.size - rnd
            pad_h = self.size - rnd
            pad_left = torch.randint(0, pad_w, (1,), dtype=torch.int).item()
            pad_right = pad_w - pad_left
            pad_top = torch.randint(0, pad_h, (1,), dtype=torch.int).item()
            pad_bottom = pad_h - pad_top
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
            return x
        else:
            return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm=nn.InstanceNorm2d, affine=True, track=False,
                 use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm, affine, track, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm, affine, track, use_dropout, use_bias):
        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm(dim, affine=affine, track_running_stats=track),
                       nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm(dim, affine=affine, track_running_stats=track)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm=nn.InstanceNorm2d, affine=True, track=False,
                 use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = (norm == nn.InstanceNorm2d)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm(ngf, affine=affine, track_running_stats=track),
                 nn.ReLU(True)]

        # downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm(ngf * mult * 2, affine=affine, track_running_stats=track),
                      nn.ReLU(True)]

        # residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm=norm, affine=affine, track=track,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        # upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm(int(ngf * mult / 2), affine=affine, track_running_stats=track),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def init_weights(m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Norm2d') != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)
