from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from sklearn.model_selection import KFold
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchaudio
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib
from torch import nn
from torch.autograd import Variable
import torchvision
import torch.distributions as tdis
import os

plt.style.use('ggplot')
np.seterr(divide='ignore', invalid='ignore')

def normalize(x):
    return -1 + 2*(x - np.min(x))/(np.max(x) - np.min(x))

class Dataset1x2_Rec(torch.utils.data.Dataset):
    def __init__(self, indexes, subjects, scenarios, subject_to_sample, test=False):
        self.indexes = indexes
        self.subjects = subjects
        self.scenarios = scenarios
        self.subject_to_sample = subject_to_sample

    def normalize(self, x):
        return -1 + 2*(x - np.min(x))/(np.max(x) - np.min(x))

    def normalize_(self, x):
        return (x - np.min(x))/(np.max(x) - np.min(x))

    def create_features(self, x):
        x = np.nan_to_num(self.normalize(x))
        x = torch.from_numpy(x).unsqueeze(dim=0).float()
        stft_f = torchaudio.transforms.Spectrogram(n_fft=255, win_length=8, power=None)

        x_spec = stft_f(x)
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()
        spec_mag = F.pad(spec_mag, pad=(6, 6), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(6, 6), mode='constant', value=0)
        ecg = torch.cat([spec_mag, spec_ang], dim=0)

        return ecg

    def __getitem__(self, index):
        segments = np.load(self.indexes[index].replace('/media/PPGECG/', '').replace('DataRandV2', 'Data')).astype(float)
        x1, x2 = segments[0, ], segments[1, ]
        x1, x2 = self.create_features(x1), self.create_features(x2)
        
        s = self.subjects[index]
        subjects = list(range(17))
        subjects.remove(s)

        sq = np.random.choice(subjects, 1)[0]
        x_sq = np.random.choice(self.subject_to_sample[sq], 1)[0]
        x_sq = np.load(x_sq.replace('/media/PPGECG/', '').replace('DataRandV2', 'Data')).astype(float)

        x_sq = x_sq[0, ]
        x_sq = self.create_features(x_sq)

        return x1, x2, x_sq, self.subjects[index], self.scenarios[index], sq

    def __len__(self):
        return self.indexes.shape[0]

class UnetDown1x2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, normalize=True, dropout=0.0, prop=False):
        super(UnetDown1x2, self).__init__()
        if prop:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, (3 if stride ==1 else 4, 4), (stride, 2), (1, 1), bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append((nn.Dropout(dropout)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp1x2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dropout=0.0, normalize=True, prop=False):
        super(UnetUp1x2, self).__init__()
        if prop:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, (3 if stride == 1 else 4), (stride, 2), 1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_sk, skip_drop):
        x = self.model(x)
        if skip_drop:
            x_sk = F.dropout(x_sk, p=0.7)
        x = torch.cat([x, x_sk], 1)
        return x

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.down3 = UnetDown1x2(24, 32)
        self.down4 = UnetDown1x2(32, 64, dropout=0.5, prop=True)
        self.down5 = UnetDown1x2(64, 64, dropout=0.5, prop=True)
        self.down6 = UnetDown1x2(64, 72, dropout=0.5)
        self.down7 = UnetDown1x2(72, 84, dropout=0.5, prop=True)
        self.down8 = UnetDown1x2(84, 84, dropout=0.5, prop=True)
        self.down9 = UnetDown1x2(84, 96, dropout=0.5)
        self.down10 = UnetDown1x2(96, 96, dropout=0.5)
        self.down11 = UnetDown1x2(96, 108, dropout=0.5, normalize=False)
        self.latent = nn.Conv2d(108, 108, kernel_size=1, padding=0)
    def forward(self, x):
        d3 = self.down3(x)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        d9 = self.down9(d8)
        d10 = self.down10(d9)
        z = self.down11(d10)
        z = self.latent(z).flatten(start_dim=1)
        z = F.normalize(z, dim=0)
        return z, (d3, d4, d5, d6, d7, d8, d9, d10)

class SkipEncode(nn.Module):
    def __init__(self):
        super(SkipEncode, self).__init__()
        self.down3 = UnetDown1x2(24, 32)
        self.down4 = UnetDown1x2(32, 64, dropout=0.5, prop=True)
        self.down5 = UnetDown1x2(64, 64, dropout=0.5, prop=True)
        self.down6 = UnetDown1x2(64, 72, dropout=0.5)
        self.down7 = UnetDown1x2(72, 84, dropout=0.5, prop=True)
        self.down8 = UnetDown1x2(84, 84, dropout=0.5, prop=True)
        self.down9 = UnetDown1x2(84, 96, dropout=0.5)
        self.down10 = UnetDown1x2(96, 96, dropout=0.5)
        self.down11 = UnetDown1x2(96, 108, dropout=0.5, normalize=False)
        self.latent = nn.Conv2d(108, 108, kernel_size=1, padding=0)

        self.mlp_fuse = nn.Sequential(
            nn.Linear(8*108, 108*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(108*4, 108*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(108*4, 108*3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(108*3, 108*2),
        )

        self.delta_gen = nn.Sequential(
            nn.Linear(108*6, 108*3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(108*3, 108*3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(108*3, 108*2),
        )

    def forward(self, x, z_p, z_t):
        d3 = self.down3(x)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        d9 = self.down9(d8)
        d10 = self.down10(d9)
        z = self.down11(d10)
        
        z = self.latent(z).flatten(start_dim=1)

        z_t = F.normalize(self.mlp_fuse(torch.cat([z, z_p, z_t], dim=-1)), dim=0)
        z_delta = F.normalize(self.delta_gen(torch.cat([z, z_p], dim=-1)), dim=0)
        
        return z_t, z_delta, (d3, d4, d5, d6, d7, d8, d9, d10)

class SkipDecode(nn.Module):
    def __init__(self):
        super(SkipDecode, self).__init__()
        self.up0 = UnetUp1x2(108, 96, dropout=0.5, normalize=False)
        self.up1 = UnetUp1x2(96 * 2, 96, dropout=0.5)
        self.up2 = UnetUp1x2(96 * 2, 84, dropout=0.5)
        self.up3 = UnetUp1x2(84 * 2, 84, dropout=0.5, prop=True)
        self.up4 = UnetUp1x2(84 * 2, 72, dropout=0.5, prop=True)
        self.up5 = UnetUp1x2(72 * 2, 64, dropout=0.5)
        self.up6 = UnetUp1x2(64 * 2, 64, dropout=0.5, prop=True)
        self.up7 = UnetUp1x2(64 * 2, 32, dropout=0.5, prop=True)
    def forward(self, z, skip, skip_drop=False):
        d3, d4, d5, d6, d7, d8, d9, d10 = skip
        u0 = self.up0(z.view(-1, 108, 1, 4), d10, skip_drop)
        u1 = self.up1(u0, d9, skip_drop)
        u2 = self.up2(u1, d8, skip_drop)
        u3 = self.up3(u2, d7, skip_drop)
        u4 = self.up4(u3, d6, skip_drop)
        u5 = self.up5(u4, d5, skip_drop)
        u6 = self.up6(u5, d4, skip_drop)
        u7 = self.up7(u6, d3, skip_drop)
        return u7

class GeneraterUNet1x2_v3(torch.nn.Module):
    def __init__(self):
        super(GeneraterUNet1x2_v3, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv2d(12, 16, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv2d(16, 24, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
        )
        self.latent = Encode()
        self.skip_latent = SkipEncode()

        self.decode_main = SkipDecode()

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(32, 24, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(24, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(8, 2, kernel_size=3, padding=1),
        )
        self.subject = nn.Linear(108*2, 17)

    def forward(self, x, xt):
        feature = self.feature(x)
        z, skips = self.latent(feature)
        z_p, z_t = torch.split_with_sizes(z, split_sizes=[108*2, 108*2], dim=-1)

        # t+1 sample
        feature_ = self.feature(xt)
        z_, skips_ = self.latent(feature_)
        z_p_, z_t_ = torch.split_with_sizes(z_, split_sizes=[108*2, 108*2], dim=-1)

        # reconstruct
        rec_t = self.reconstruct(self.decode_main(z, skips, skip_drop=True))
        rec_t_ = self.reconstruct(self.decode_main(z_, skips, skip_drop=True))

        p_pred = F.softmax(self.subject(z_p), dim=-1)

        return (rec_t, rec_t_), z_p, z_t, z_p_, z_t_, feature, z, p_pred, skips, skips_
        
    def decode(self, z, d3, d4, d5, d6, d7, d8, d9, d10):
        return self.reconstruct(self.decode_main(z, d3, d4, d5, d6, d7, d8, d9, d10, skip_drop=True))

class SubjectClassifier(torch.nn.Module):
    def __init__(self):
        super(SubjectClassifier, self).__init__()

        self.feature = nn.Sequential(
            self.conv_layer(2, 4, stride=1, normalize=False),
            self.conv_layer(4, 8, down=False),
            self.conv_layer(8, 12, down=False),
            self.conv_layer(12, 16, down=False),
            self.conv_layer(16, 24),
            self.conv_layer(24, 32),
            self.conv_layer(32, 48, down=False),
        )

        self.embedding = torch.nn.Sequential(
            self.conv_layer(48, 64, dropout=0.5),
            self.conv_layer(64, 72, dropout=0.5),
            self.conv_layer(72, 84, dropout=0.5),
            self.conv_layer(84, 84, dropout=0.5),
            nn.Conv2d(84, 84, 4, 2, 1),
        )
        self.subject = nn.Linear(168, 17)

    def conv_layer(self, in_channel, out_channel, stride=2, normalize=True, dropout=0.0, down=True):
        if down:
            layers = [nn.Conv2d(in_channel, out_channel, (3 if stride == 1 else 4, 4), (stride, 2), (1, 1), bias=False)]
        else:
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        f = self.feature(x)
        z = self.embedding(f).flatten(start_dim=1)
        return torch.softmax(self.subject(z), dim=-1)

class XXQ_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=(3, 7), padding=(1, 3), stride=(4, 4)),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(24, 36, kernel_size=(3, 7), padding=(1, 3), stride=(4, 4)),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(36, 64, kernel_size=(3, 7), padding=(1, 3), stride=(4, 8)),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, x, xq):
        x = torch.cat([x, xq], dim=1)
        return self.model(x).squeeze(dim=-1)

class Discrim_MLP(nn.Module):
    def __init__(self):
        super(Discrim_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(108*4, 256),
            nn.LeakyReLU(0.4),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x)

def MMD(x, y, kernel='rbf'):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to('cuda:1'),
                  torch.zeros(xx.shape).to('cuda:1'),
                  torch.zeros(xx.shape).to('cuda:1'))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx.to('cuda:1') / a)
            YY += torch.exp(-0.5 * dyy.to('cuda:1') / a)
            XY += torch.exp(-0.5 * dxy.to('cuda:1') / a)

    return torch.mean(XX + YY - 2. * XY)


def evaluate_l1_loss(y_hat, y, ref, alpha=0.5):  
    ref[ref == 1.0] = alpha
    ref[ref == 0.0] = 1.0 - alpha
    loss = F.l1_loss(y, y_hat.squeeze(dim=1), reduction='none') * ref
    return loss.mean(dim=-1).mean()*10

def transform(x):
    inv_spec = torchaudio.transforms.InverseSpectrogram(n_fft=255, win_length=8)
    real_m, real_a = x[:,0, :,  6:506], x[:,1, :, 6:506]
    z_real = torch.complex(real_m * torch.cos(real_a) , real_m * torch.sin(real_a))
    return inv_spec(z_real.cpu()).cpu().detach().numpy()

def cosinesim(x1, x2):
    return torch.nn.CosineSimilarity()(x1, x2).mean()

class DRecons(nn.Module):
    def __init__(self, in_chan=2):
        super(DRecons, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, 12, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(12, 24, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(24, 36, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(36),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(36, 48, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(48, 64, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(64, 72, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(72),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(72, 96, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(96*4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.encoder(x)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

# data loader
import pickle5 as pickle
recording_to_files = None
with open(r"./Data/recording_to_segment.pickle", "rb") as input_file:
    recording_to_files = pickle.load(input_file)
from sklearn.model_selection import train_test_split
ignore = ['ECGPCG00{:02d}'.format(k) for k in [3, 4, 5, 6, 7, 8, 9, 40]]

import pandas as pd 
df = pd.read_csv('ECGPCGSpreadsheet_mod.csv')

sub_to_id = {}
subjects = ['S023', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S001', 'S010', 'S002', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S011', 'S020']
for i in range(len(subjects)):
    sub_to_id[subjects[i]] = i

N = 10000
Y_train = []
X_train = []
S_train = []
for record, condition, clss, sid in zip(df['Record Name'], df['ECG Notes'], df['Class'], df['Subject ID']):
    if condition == 'Good' and record not in ignore:
        X_train += np.random.choice(recording_to_files[record], size=N, replace=False).tolist()
        Y_train += [clss] * N
        S_train += [sub_to_id[sid]] * N

X_train = np.array(X_train)
Y_train = np.array(Y_train)
S_train = np.array(S_train)

S_train[S_train == 18] = 1
S_train[S_train == 19] = 2
S_train[S_train == 20] = 5
S_train[S_train == 17] = 6
print('train on: (1120000)', X_train.shape, Y_train.shape, 'we are using:', 1120000/X_train.shape[0])

# subject-to-sample
subject_to_sample = {}
for i in range(17):
    idx = np.where(S_train == i)[0]
    subject_to_sample[i] = X_train[idx]

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}

training_set = Dataset1x2_Rec(X_train, S_train, Y_train, subject_to_sample)
training_generator = torch.utils.data.DataLoader(training_set, **params)

model = GeneraterUNet1x2_v3()
model.to('cuda:1')

subjectclf = SubjectClassifier()
subjectclf.to('cuda:1')
subjectclf.load_state_dict(torch.load('./checkpoints/model_classifier.tm'))
freeze(subjectclf)

discrim = Discrim_MLP()
discrim.to('cuda:1')

discrimre = DRecons()
discrimre.to('cuda:1')

z = 128+64
dst = tdis.MultivariateNormal(loc=torch.zeros((z,)), covariance_matrix=torch.eye(z))

import itertools
enc_optim = torch.optim.Adam(itertools.chain(
    model.feature.parameters(),
    model.latent.parameters(),
    model.subject.parameters()
    ), lr=0.0001)

dec_optim = torch.optim.Adam(itertools.chain(
    model.decode_main.parameters(),
    model.reconstruct.parameters())
    , lr=0.0001)

shift_optim = torch.optim.AdamW(itertools.chain(
    model.skip_latent.parameters(),
    ), lr=0.0001)

discrim_optim = torch.optim.AdamW(discrim.parameters(), lr=0.0001)
reconsd_optim = torch.optim.AdamW(discrimre.parameters(), lr=0.0001)

DIR = 'T9/V14'
code = 'waegan-{}-adversarial'.format(DIR.replace('/', '-'))
min_loss = -np.inf
last_improvmnt = -1

epoch_train, epoch_test = [], []
generat_loss, discrimin_loss, distrib_loss, shrink_loss = [], [], [], []

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def cos_loss(x1, x2):
    return (1 - cos(x1, x2)).mean()

adversarial_loss = F.binary_cross_entropy

dirname = './SIN/checkpoint'

# we load-v14, we don't have discrim-re values with L_D(adv-re), relaxed
try:
    model.load_state_dict(torch.load('{}/model_{}.tm'.format(dirname, code)))
    discrim.load_state_dict(torch.load('{}/discrim_{}.tm'.format(dirname, code)))

    enc_optim.load_state_dict(torch.load('{}/enc_opt_{}.tm'.format(dirname, code)))
    dec_optim.load_state_dict(torch.load('{}/dec_opt_{}.tm'.format(dirname, code)))
    discrim_optim.load_state_dict(torch.load('{}/discrim_opt_{}.tm'.format(dirname, code)))
    shift_optim.load_state_dict(torch.load('{}/shift_opt_{}.tm'.format(dirname, code)))
    
    print('loaded the saved models at:', dirname, code)

except Exception as e:
    print('faild loading the saved models at:', dirname, e)
model.load_state_dict(torch.load('{}/model_{}.tm'.format(dirname, code)))

#-re-run
DIR = 'T4' # this is the v14 and we removed the L_D-re epochss, also let's focus on the final generation itself with modification for the conditional generation as well! 
code = 'waegan-{}-adversarial'.format(DIR.replace('/', '-'))

try:
    os.mkdir('./SIN/{}'.format(DIR))
except Exception as e:
    print(e)
    pass

for epoch in range(1000):
    print('---Epoch---{:d}-{}'.format(epoch, code))
    model.train()
    train_loss, train_acc, g_loss, d_loss, mmd_loss_agg, shrink = [], [], [], [], [], []

    for i, (xt, xt_p1, x_sq, s, sc, sq) in enumerate(training_generator):        
        enc_optim.zero_grad()
        dec_optim.zero_grad()
        discrim_optim.zero_grad()
        reconsd_optim.zero_grad()

        valid = Variable(torch.from_numpy(np.ones((64, 1))), requires_grad=False).float().to('cuda:1')
        fake = Variable(torch.from_numpy(np.zeros((64, 1))), requires_grad=False).float().to('cuda:1')
        
        (rec_t, rec_t_), z_p, z_t, z_p_, z_t_, feature, z, p_pred, skips, skips_  = model(xt.to('cuda:1'), xt_p1.to('cuda:1'))
        
        # classificaiton
        clf_loss = F.cross_entropy(p_pred, s.to('cuda:1'))
        f1s = f1_score(s.cpu().detach().numpy(), p_pred.argmax(dim=-1).cpu().detach().numpy(), average='micro')
        bs = xt.shape[0]

        recons_loss = F.mse_loss(rec_t, xt.to('cuda:1')) + F.mse_loss(rec_t_, xt_p1.to('cuda:1'))

        loss = recons_loss + clf_loss 
        loss.backward(retain_graph=True)

        # conditional-generation steps
        freeze(model.decode_main)
        freeze(model.reconstruct)

        # conditional-generate
        crec_tp_1 = model.reconstruct(model.decode_main(torch.cat([z_p, z_t_], dim=-1), skips))
        crec_t = model.reconstruct(model.decode_main(torch.cat([z_p_, z_t], dim=-1), skips_))

        recons_loss_cond = F.mse_loss(crec_tp_1, xt_p1.to('cuda:1')) + F.mse_loss(crec_t, xt.to('cuda:1')) +\
              0.5*(adversarial_loss(discrimre(crec_tp_1), valid) + 0.5*adversarial_loss(discrimre(crec_t), valid))
        
        recons_loss_cond.backward(retain_graph=True)

        # estimate-the-shift
        shift_optim.zero_grad()

        # freeze the feature encoder
        freeze(model.feature)

        z_t_shift, delta, skips_lat = model.skip_latent(feature.detach(), z_p, z_t)
        #shift_loss_latent = 100*F.mse_loss(z_t_shift, z_t_.detach())
        shift_loss_latent = cos_loss(z_t_shift, z_t_.detach())

        x_recons_shift = model.reconstruct(model.decode_main(torch.cat([z_p.detach() + delta, z_t_shift], dim=-1), skips_lat))
        shift_rec_loss = F.mse_loss(x_recons_shift, xt_p1.to('cuda:1'))

        adver_loss = adversarial_loss(discrim(torch.cat([z_t_shift, z_t], dim=-1)), valid)
        adver_loss_rec = adversarial_loss(discrimre(x_recons_shift), valid)

        shift_loss = 0.5*(shift_loss_latent + adver_loss) + shift_rec_loss + adver_loss_rec # 0.5(latent) + final-re
        shift_loss.backward(retain_graph=True)
        shift_optim.step()

        with torch.no_grad():# subject
            s_pred = subjectclf(x_recons_shift)
            f1_gen = f1_score(s.detach().numpy(), s_pred.argmax(dim=-1).detach().cpu().numpy(), average='micro')
            conf = confusion_matrix(s.detach().numpy(), s_pred.argmax(dim=-1).detach().cpu().numpy())

            s_pred = subjectclf(xt_p1.to('cuda:1'))
            f1_real = f1_score(s.detach().numpy(), s_pred.argmax(dim=-1).detach().cpu().numpy(), average='micro')

        # freeze the feature encoder
        unfreeze(model.feature)

        enc_optim.step()
        dec_optim.step()

        unfreeze(model.decode_main)
        unfreeze(model.reconstruct)

        # discriminator-loss
        discrim_loss = 0.5*(adversarial_loss(discrim(torch.cat([z_t_shift, z_t], dim=-1).detach()), fake) +\
                                    adversarial_loss(discrim(torch.cat([z_t_, z_t], dim=-1).detach()), valid))

        discrim_loss.backward()
        discrim_optim.step()

        rediscrim_loss = 0.5*(adversarial_loss(discrimre(xt_p1.to('cuda:1')), valid) + adversarial_loss(discrimre(x_recons_shift.detach()), fake))
        rediscrim_loss.backward()
        reconsd_optim.step()

        g_loss.append(recons_loss.item())
        d_loss.append(recons_loss_cond.item())
        shrink.append([clf_loss.item(), shift_loss_latent.item(), shift_rec_loss.item(), 
                        adver_loss.item(), discrim_loss.item(), rediscrim_loss.item(), adver_loss_rec.item(), f1_gen])
 
        print('\r[{:4d}]train loss: {:.4f} recons: {:.4f}|{:.4f} f1: {:.4f} shift_loss: {:.4f}|{:.4f}  G():{:.4f} D:{:.4f} z: {:.4f}|{:.4f} red: {:.4f}|{:.4f}'.format(i, 
            loss.item(), 
            recons_loss.item(), 
            recons_loss_cond.item(), 
            f1s, 
            shift_loss.item(), shift_loss_latent.item(),
            adver_loss.item(), discrim_loss.item(),
            torch.min(z), torch.max(z),
            rediscrim_loss.item(), adver_loss_rec.item()
            ), end='')
        
        if i in [10, 100, 500, 1000]:
            dirname = './SIN/{}/f_{:d}'.format(DIR, epoch)
            import os 
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass
            xt = transform(xt)
            xt_p1 = transform(xt_p1)

            xt_recons = transform(rec_t)
            xt_p1_recons = transform(rec_t_)
            xt_p1_recons_cond = transform(crec_tp_1)
            xt_p1_recons_shift = transform(x_recons_shift)
            
            for k in range(xt.shape[0]):
                _ = plt.figure(figsize=(12, 3))

                plt.subplot(141)
                plt.plot(xt[k, ], 'b', alpha=0.5)
                plt.plot(xt_recons[k, ], 'g')
                plt.title('$x_t$')

                plt.subplot(142)
                plt.plot(xt_p1[k, ], 'b', alpha=0.5)
                plt.plot(xt_p1_recons[k, ], 'b')
                plt.title('$x_{t+1}$')

                plt.subplot(143)
                plt.plot(xt_p1[k, ], 'b', alpha=0.5)
                plt.plot(xt_p1_recons_cond[k, ], 'b')
                plt.title('$x_{t+1}|zp_t$')

                plt.subplot(144)
                plt.plot(xt_p1[k, ], 'b', alpha=0.5)
                plt.plot(xt_p1_recons_shift[k, ], 'b')
                plt.title('$x_{t+1}|sh$')
                
                plt.tight_layout()
                plt.savefig('{}/F_{:d}.png'.format(dirname, k))
                plt.close()
                plt.clf()
            #break

    print('\n---({}) L_rec: {:.4f} L_crec: {:.4f}'.format(code, np.mean(g_loss), np.mean(d_loss))) 

    plt.plot(d_loss)
    plt.savefig('zdist.png')
    plt.clf()
    plt.close()

    generat_loss.append(np.mean(g_loss))
    discrimin_loss.append(np.mean(d_loss))

    shrink_loss_ = np.array(shrink)
    shrink_loss_ = shrink_loss_.mean(axis=0)
    shrink_loss.append(shrink_loss_)

    temp = np.array(shrink_loss)
    _ = plt.figure(figsize=(12, 9))

    plt.subplot(431)
    plt.plot(generat_loss)
    plt.title('$L_{re}$')

    plt.subplot(432)
    plt.plot(discrimin_loss)
    plt.title('$L_{re-c}$')

    plt.subplot(434)
    plt.plot(temp[:, 0])
    plt.title('$L_{cce}$')

    plt.subplot(435)
    plt.plot(temp[:, 1])
    plt.title('$L_{sh-mse}$')

    plt.subplot(436)
    plt.plot(temp[:, 2])
    plt.title('$L_{sh-re}$')

    plt.subplot(437)
    plt.plot(temp[:, 3])
    plt.title('$L_{G}$')

    plt.subplot(438)
    plt.plot(temp[:, 4])
    plt.title('$L_{D}$')

    plt.subplot(439)
    plt.plot(temp[:, 5])
    plt.title('$L_{r-D}$')

    plt.subplot(4,3,10)
    plt.plot(temp[:, 6])
    plt.title('$L_{r-G}$')

    plt.subplot(4,3,11)
    plt.plot(temp[:, 7])
    plt.title('$f1_{s}$')

    plt.tight_layout()
    plt.savefig('./Loss/loss-{}.png'.format(code))
    plt.clf()
    plt.close()

    # save models
    # save models
    torch.save(model.state_dict(), '{}/model_{}.tm'.format(dirname, code))
    torch.save(discrim.state_dict(), '{}/discrim_{}.tm'.format(dirname, code))
    torch.save(discrimre.state_dict(), '{}/discrimre_{}.tm'.format(dirname, code))
    
    torch.save(enc_optim.state_dict(), '{}/enc_opt_{}.tm'.format(dirname, code))
    torch.save(dec_optim.state_dict(), '{}/dec_opt_{}.tm'.format(dirname, code))
    torch.save(discrim_optim.state_dict(), '{}/discrim_opt_{}.tm'.format(dirname, code))
    torch.save(shift_optim.state_dict(), '{}/shift_opt_{}.tm'.format(dirname, code))

'''
epoch = 50
dirname = './SIN/{}/f_{:d}'.format(DIR, epoch)
model.load_state_dict(torch.load('{}/model_{}.tm'.format(dirname, code)))

N = 10
processed = [0]*17

# evaluate the modals
for i, (xt, xt_p1, x_sq, s, sc, sq) in enumerate(training_generator): 
    (rec_t, _), z_p, z_t, _, _, feature, _, _, _, _  = model(xt.to('cuda:1'), xt_p1.to('cuda:1'))
    z_t_shift, delta, skips_lat = model.skip_latent(feature.detach(), z_p, z_t)    
    x_recons_shift = model.reconstruct(model.decode_main(torch.cat([z_p.detach() + delta, z_t_shift], dim=-1), skips_lat))

    # reconstruct
    xt_recons = transform(rec_t)
    xt_p1_recons_shift = transform(x_recons_shift)

    # main-signals
    xt = transform(xt)
    xt_p1 = transform(xt_p1)

    for idx, subj in enumerate(s):
        if processed[subj.item()] < N:
            _ = plt.figure(figsize=(8, 3))
            
            plt.subplot(121)
            plt.plot(xt[idx, ], 'g')
            plt.plot(xt_recons[idx, ], 'b')

            plt.subplot(122)
            plt.plot(xt_p1[idx, ], 'g')
            plt.plot(xt_p1_recons_shift[idx, ], 'b')

            dirname = './Generations/V4/f{:02d}'.format(subj.item())
            import os 
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass

            plt.tight_layout()
            plt.savefig('./Generations/V4/f{:02d}/eval_{:d}_{:d}.png'.format(subj.item(), subj.item(), processed[subj.item()]))

            plt.clf()
            plt.close()

            print('processed:', './Generations/f{:02d}/eval_{:d}_{:d}.png'.format(subj.item(), subj.item(), processed[subj.item()]))
            processed[subj.item()] += 1

exit()

model.load_state_dict(torch.load('{}/model_{}.tm'.format(dirname, code)))

# latent-per-subject
L1, L2, L3, S = [], [], [], []

for i, (x, ref, s, sc) in enumerate(training_generator):
    z_p, z_sc, z_res, p_pred, sc_pred = model.encoder(x.to('cuda:1'))
    L1.append(z_p.cpu().detach())
    L2.append(z_sc.cpu().detach())
    L3.append(z_res.cpu().detach())
    S.append(s.cpu().detach())
    if i == 100:
        break

L1 = torch.cat(L1, dim=0)
L2 = torch.cat(L2, dim=0)
L3 = torch.cat(L3, dim=0)
S = torch.cat(S, dim=0)

for i, (x, ref, s, sc) in enumerate(training_generator):        
    # train-the-generator
    z_p, z_sc, z_res, p_pred, sc_pred = model.encoder(x.to('cuda:1'))

    # for different subjects
    for i in range(16):
        idx = torch.where(S == i)[0]
        plt.figure(figsize=(8, 4))
        for z in range(6):
            z_p_ = L1[idx[z, ]].to('cuda:1')
            x_recons = model.decoder(torch.cat([z_p_, z_sc[0], z_res[0]], dim=-1).unsqueeze(dim=1))
            x_recons = transform(x_recons)

            plt.subplot(3,2,z+1)
            plt.plot(x_recons[0])
            plt.title('{}-from-{}'.format(s[0].item(), i))

        plt.tight_layout()
        plt.savefig('./SubjectVary/subject-{}.png'.format(i))

    exit()

Z = np.concatenate(Z, axis=0)
Y = np.concatenate(Y, axis=0)
S = np.concatenate(S, axis=0)

print(np.unique(Y))
print(Y.shape, Z.shape)

# tSNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, verbose=1, perplexity=33, n_iter=10000, learning_rate=100).fit_transform(Z)

plt.style.use('ggplot')

_ = plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title('scenario-info')
for i in range(6):
    x_ = X_embedded[Y == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], s=8, label='p-{}'.format(i))
plt.legend()

col = ['red', 'maroon', 'tan', 'lime', 'darkgreen', 'peru', 'olive', 'gold', 'cyan', 'teal', 'navy', 'indigo', 'purple', 'pink', 'yellow', 'green', 'grey', 'brown']

plt.subplot(122)
plt.title('subject-info')
for i in range(17):
    x_ = X_embedded[S == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], s=8, c=col[i], label='p-{}'.format(i))
plt.legend()

plt.savefig('zp-tsne-{}.png'.format(DIR.replace('/', '')))

exit()


z = []
for i, (xt, xt_p1, s, sc) in enumerate(training_generator):        
    z_p, z_sc, z_res, p_pred, sc_pred = model.encoder(xt.to('cuda:1'))
    z.append(z_res.detach().cpu())
    if i == 100:
        break
z = torch.cat(z, dim=0).numpy()
print(z.shape)

plt.figure(figsize=(8, 8))
for k, i in enumerate(range(0, 56, 7)):
    plt.subplot(4,2,k+1)
    plt.boxplot(z[:, i:i+7])
    plt.title('z[{:d}, {:d}]'.format(i,i+7))

plt.tight_layout()
plt.savefig('boxplot-{}.png'.format(DIR.replace('/', '-')))

exit()

import itertools

enc_optim = torch.optim.Adam(itertools.chain(model.feature_encoder.parameters(), model.disentangle.parameters()), lr=0.001)
dec_optim = torch.optim.Adam(model.decode.parameters(), lr=0.001)
clasf_optim = torch.optim.Adam(itertools.chain(model.device.parameters(), model.pathology.parameters()), lr=0.0001)


# tsnes
Z = []
Y = []
S = []

for i, (x, ref, s, sc) in enumerate(training_generator):        
    shape = (x.shape[0], 12)
    valid = Variable(torch.from_numpy(np.ones(shape)), requires_grad=False).float().to('cuda:1')
    fake = Variable(torch.from_numpy(np.zeros(shape)), requires_grad=False).float().to('cuda:1')

    # train-the-generator
    z_p, z_sc, z_res, p_pred, sc_pred = encoder(x.to('cuda:1'))

    Z.append(z_res.cpu().detach().numpy())
    Y.append(sc.detach().numpy())
    S.append(s.detach().numpy())

    if i == 100:
        break

Z = np.concatenate(Z, axis=0)
Y = np.concatenate(Y, axis=0)
S = np.concatenate(S, axis=0)

print(np.unique(Y))
print(Y.shape, Z.shape)

# tSNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, verbose=1, perplexity=33, n_iter=10000, learning_rate=100).fit_transform(Z)

plt.style.use('ggplot')

_ = plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title('scenario')
for i in range(6):
    x_ = X_embedded[Y == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], s=8, label='p-{}'.format(i))
plt.legend()

plt.subplot(122)
plt.title('subject-info')
for i in range(10):
    x_ = X_embedded[S == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], s=8, label='p-{}'.format(i))
plt.legend()

plt.savefig('residual-tsne.png')

exit()

model.load_state_dict(torch.load('./Models/model_{}.tm'.format(code)))
#t-SNEs
def transform(x):
    inv_spec = torchaudio.transforms.InverseSpectrogram(n_fft=255, win_length=8)
    real_m, real_a = x[:,0, :,  6:506], x[:,1, :, 6:506]
    z_real = torch.complex(real_m * torch.cos(real_a) , real_m * torch.sin(real_a))
    return inv_spec(z_real.cpu()).cpu().detach().numpy()

Z = []
Y = []
S = []

# generate a base latent vector
for index in np.random.choice(X_train.shape[0], 100, replace=False):
    x = np.load(X_train[index]).astype(float)[0, ]
    x = np.nan_to_num(normalize(x))

    x = torch.from_numpy(x).unsqueeze(dim=0).float()
    stft_f = torchaudio.transforms.Spectrogram(win_length=8, n_fft=255, power=None)

    x_spec = stft_f(x)
    spec_mag, spec_ang = x_spec.abs(), x_spec.angle()
    spec_mag = F.pad(spec_mag, pad=(6, 6), mode='constant', value=0)
    spec_ang = F.pad(spec_ang, pad=(6, 6), mode='constant', value=0)
    ecg = torch.cat([spec_mag, spec_ang], dim=0)

    x = ecg.unsqueeze(dim=0)
    z_p, z_sc, z_res, p_pred, sc_pred = model.encoder(x.to('cuda:1'))
    x_recons = model.decoder(torch.cat([z_p, z_sc, z_res], dim=-1))

    _ = plt.figure(figsize=(12, 12))
    plt.subplot(8,3,1)
    plt.plot(transform(x)[0, ], 'g', alpha=0.5)
    plt.plot(transform(x_recons)[0, ], 'b')

    # vary-a-latent-dim
    for i in range(23):
        z_res = d3.sample(sample_shape=(1,)).to('cuda:1')
        x_recons = model.decoder(torch.cat([z_p, z_sc, z_res], dim=-1))

        plt.subplot(8,3,i+2)
        plt.plot(transform(x_recons)[0, ])
        plt.plot(transform(x)[0, ], 'g', alpha=0.4)
    plt.tight_layout()

    plt.savefig('./TimeShift/vary_z_res_{:d}.png'.format(index))
    plt.clf()
    plt.close()

exit()
'''

'''
try:
    #encoder.load_state_dict(torch.load('./Models/enc_{}.tm'.format(code)))
    #decoder.load_state_dict(torch.load('./Models/dec_{}.tm'.format(code)))
    #discriminator.load_state_dict(torch.load('./Models/dis_{}.tm'.format(code)))
    model.load_state_dict(torch.load('./Models/model_{}.tm'.format(code)))
    
    enc_optim.load_state_dict(torch.load('./Models/enc_opt_{}.tm'.format(code)))
    dec_optim.load_state_dict(torch.load('./Models/dec_opt_{}.tm'.format(code)))
    #dis_optim.load_state_dict(torch.load('./Models/dis_opt_{}.tm'.format(code)))
    print('sucessfully loaded the saved configurations.')
except Exception:
    print('error...failed to load the save dicts')

train_dis = False

model.load_state_dict(torch.load('./Models/model_{}.tm'.format(code)))

# tsnes
Z = []
Y = []
S = []

for i, (x, ref, s, sc) in enumerate(training_generator):        
    shape = (x.shape[0], 12)
    valid = Variable(torch.from_numpy(np.ones(shape)), requires_grad=False).float().to('cuda:1')
    fake = Variable(torch.from_numpy(np.zeros(shape)), requires_grad=False).float().to('cuda:1')

    # train-the-generator
    z_p, z_sc, z_res, p_pred, sc_pred = model.encoder(x.to('cuda:1'))

    _ = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.title('$z_{p}$')
    im = plt.imshow(z_p.detach().cpu().numpy())
    plt.colorbar(im, orientation="horizontal", pad=0.1)


    plt.subplot(132)
    plt.title('$z_{sc}$')
    im = plt.imshow(z_sc.detach().cpu().numpy())
    plt.colorbar(im, orientation="horizontal", pad=0.1)

    plt.subplot(133)
    plt.title('$z_{res}$')
    im = plt.imshow(z_res.detach().cpu().numpy())
    plt.colorbar(im, orientation="horizontal", pad=0.1)

    plt.tight_layout()
    plt.savefig('./Latent/V2/f_{:d}.png'.format(i))
    plt.clf()
    plt.close()

    Z.append(z_res.cpu().detach().numpy())
    Y.append(sc.detach().numpy())
    S.append(s.detach().numpy())

    if i == 100:
        break


exit()

Z = np.concatenate(Z, axis=0)
Y = np.concatenate(Y, axis=0)
S = np.concatenate(S, axis=0)

print(np.unique(S))
print(Y.shape, Z.shape)

# tSNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, verbose=1, perplexity=33, n_iter=10000, learning_rate=100).fit_transform(Z)

plt.style.use('ggplot')
col = ['red', 'maroon', 'tan', 'lime', 'darkgreen', 'peru', 'olive', 'gold', 'cyan', 'teal', 'navy', 'indigo', 'purple', 'pink', 'yellow', 'green', 'grey', 'brown']

_ = plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title('scenario')
for i in range(6):
    x_ = X_embedded[Y == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], s=8, label='p-{}'.format(i))
plt.legend()

plt.subplot(122)
plt.title('subject')
for i in range(17):
    x_ = X_embedded[S == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], c=col[i], s=8, label='p-{}'.format(i))
plt.legend()
plt.savefig('./tSNE/z_res_{}.png'.format(code))

exit()
'''

'''

# conductance
from captum.attr import LayerConductance
params = {'batch_size': 24,
          'shuffle': True,
          'num_workers': 11}

training_set = Dataset1x2_Rec(X_train, S_train, Y_train, subject_to_sample)
training_generator = torch.utils.data.DataLoader(training_set, **params)
for epoch in range(22, 25):
    dirname = './SIN/{}/f_{:d}'.format(DIR, epoch)
    model.load_state_dict(torch.load('{}/model_{}.tm'.format(dirname, code)))

    z = []
    for i, (xt, xt_p1, x_sq, s, sc, sq) in enumerate(training_generator):
        feature = model.feature(xt.to('cuda:1'))
        z, skips = model.latent(feature)
        (d3, d4, d5, d6, d7, d8, d9, d10) = skips
        if i == 0:
            break
        print(xt.shape)

    dim = (2, 128, 512)
    z_ = torch.zeros((1, 108, 1, 4))

    model = model.to('cuda:1')
    conductance = LayerConductance(model.decode, model.decode_main.up0.model[0])
    for i in range(dim[0]):
        for j in range(0, dim[1], 16):
            for k in range(0, dim[2], 32):
                attribution = conductance.attribute((z, d3, d4, d5, d6, d7, d8, d9, d10), target=(i, j, k), attribute_to_layer_input=True)
                z_ += attribution.mean(dim=0).cpu()
                print(i, j, k)

    z_ = z/(2*128*512)
    z_ = z_.flatten(start_dim=1)

    z__ = z_.log().detach().cpu().numpy()
    z_ = z_.detach().cpu().numpy()

    plt.figure(figsize=(12, 3))

    im = plt.imshow(z_)
    plt.colorbar(im, orientation="horizontal", pad=0.1)

    plt.savefig('./Cond/V77/conductance-{}.png'.format(epoch))

exit()
'''