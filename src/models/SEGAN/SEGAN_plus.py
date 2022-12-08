""" Forked from https://github.com/santi-pdp/segan_pytorch """
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import math

def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)

# SincNet conv layer
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                                                                 -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right, cuda=False):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    ones = torch.ones(1)
    if cuda:
        ones = ones.to('cuda')
    y=torch.cat([y_left, ones, y_right])

    return y

class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs,
                 padding='VALID'):
        super(SincConv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel,
                                 N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1)) # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale=fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding

    def forward(self, x):
        cuda = x.is_cuda
        filters=torch.zeros((self.N_filt, self.Filt_dim))
        N=self.Filt_dim
        t_right=torch.linspace(1, (N - 1) / 2,
                               steps=int((N - 1) / 2)) / self.fs
        if cuda:
            filters = filters.to('cuda')
            t_right = t_right.to('cuda')

        min_freq=50.0
        min_band=50.0
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)
        n = torch.linspace(0, N, steps = N)
        # Filter window (hamming)
        window=(0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window.to('cuda')
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float()* \
                        sinc(filt_beg_freq[i].float() * self.freq_scale,
                             t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float()* \
                        sinc(filt_end_freq[i].float() * self.freq_scale,
                             t_right, cuda)
            band_pass=(low_pass2 - low_pass1)
            band_pass=band_pass/torch.max(band_pass)
            if cuda:
                band_pass = band_pass.to('cuda')

            filters[i,:]=band_pass * window
        if self.padding == 'SAME':
            x_p = F.pad(x, (self.Filt_dim // 2,
                            self.Filt_dim // 2), mode='reflect')
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim))
        return out


class GConv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kernel_size, stride=1,
                 bias=True, norm_type=None):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kernel_size, stride=stride, bias=bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, ret_linear=False):
        if self.stride > 1:
            P = (self.kernel_size // 2 - 1,
                 self.kernel_size // 2)
        else:
            P = (self.kernel_size // 2,
                 self.kernel_size // 2)
        x_p = F.pad(x, P, mode='reflect')
        a = self.conv(x_p)
        a = self.forward_norm(a, self.norm)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h

class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kernel_size, stride=4,
                 bias=True,
                 norm_type=None,
                 act=None):
        super().__init__()
        pad = max(0, (stride - kernel_size)//-2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps,
                                         kernel_size = kernel_size,
                                         stride=stride,
                                         padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     fmaps)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        h = self.deconv(x)
        if self.kernel_size % 2 != 0:
            h = h[:, :, :-1]
        h = self.forward_norm(h, self.norm)
        h = self.act(h)
        return h

class GSkip(nn.Module):

    def __init__(self, skip_type, size, skip_init, skip_dropout=0,
                 merge_mode='sum', kernel_size=11, bias=True):
        # skip_init only applies to alpha skips
        super().__init__()
        self.merge_mode = merge_mode
        if skip_type == 'alpha' or skip_type == 'constant':
            if skip_init == 'zero':
                alpha_ = torch.zeros(size)
            elif skip_init == 'randn':
                alpha_ = torch.randn(size)
            elif skip_init == 'one':
                alpha_ = torch.ones(size)
            else:
                raise TypeError('Unrecognized alpha init scheme: ',
                                skip_init)
            #if cuda:
            #    alpha_ = alpha_.cuda()
            if skip_type == 'alpha':
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
            else:
                # constant, not learnable
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
                self.skip_k.requires_grad = False
        elif skip_type == 'conv':
            if kernel_size > 1:
                pad = kernel_size // 2
            else:
                pad = 0
            self.skip_k = nn.Conv1d(size, size, kernel_size, stride=1,
                                    padding=pad, bias=bias)
        else:
            raise TypeError('Unrecognized GSkip scheme: ', skip_type)
        self.skip_type = skip_type
        if skip_dropout > 0:
            self.skip_dropout = nn.Dropout(skip_dropout)

    def __repr__(self):
        if self.skip_type == 'alpha':
            return self._get_name() + '(Alpha(1))'
        elif self.skip_type == 'constant':
            return self._get_name() + '(Constant(1))'
        else:
            return super().__repr__()

    def forward(self, hj, hi):
        if self.skip_type == 'conv':
            sk_h = self.skip_k(hj)
        else:
            skip_k = self.skip_k.repeat(hj.size(0), 1, hj.size(2))
            sk_h =  skip_k * hj
        if hasattr(self, 'skip_dropout'):
            sk_h = self.skip_dropout(sk_h)
        if self.merge_mode == 'sum':
            # merge with input hi on current layer
            return sk_h + hi
        elif self.merge_mode == 'concat':
            return torch.cat((hi, sk_h), dim=1)
        else:
            raise TypeError('Unrecognized skip merge mode: ', self.merge_mode)


class Generator(nn.Module):

    def __init__(self, input_size,
                 kernel_size,
                 dec_fmaps=None,
                 dec_kernel_size=None,
                 dec_poolings=None,
                 z_dim=1024,
                 no_z=False,
                 skip=True,
                 bias=False,
                 skip_init='one',
                 skip_dropout=0,
                 skip_type='alpha',
                 norm_type=None,
                 skip_merge='sum',
                 skip_kernel_size=11,
                 name='Generator'):

        super().__init__()
        self.name = name
        self.skip = skip
        self.poolings = [4]*5
        self.fmaps = [64, 128, 256, 512, 1024]
        self.bias = bias
        self.no_z = no_z
        self.z_dim = z_dim
        self.enc_blocks = nn.ModuleList()
        assert isinstance(self.fmaps, list), type(self.fmaps)
        assert isinstance(self.poolings, list), type(self.poolings)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(self.fmaps)
        assert isinstance(kernel_size, list), type(kernel_size)
        skips = {}
        ninp = input_size
        for pi, (fmap, pool, kw) in enumerate(zip(self.fmaps, self.poolings, kernel_size),
                                              start=1):
            if skip and pi < len(self.fmaps):
                # Make a skip connection for all but last hidden layer
                gskip = GSkip(skip_type, fmap,
                              skip_init,
                              skip_dropout,
                              merge_mode=skip_merge,
                              kernel_size=skip_kernel_size,
                              bias=bias)
                l_i = pi - 1
                skips[l_i] = {'alpha':gskip}
                setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
            enc_block = GConv1DBlock(
                ninp, fmap, kw, stride=pool, bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap

        self.skips = skips
        if not no_z and z_dim is None:
            z_dim = self.fmaps[-1]
        if not no_z:
            ninp += z_dim
        # Ensure we have fmaps, poolings and kernel_size ready to decode
        if dec_fmaps is None:
            dec_fmaps = self.fmaps[::-1][1:] + [1]
        else:
            assert isinstance(dec_fmaps, list), type(dec_fmaps)
        if dec_poolings is None:
            dec_poolings = self.poolings[:]
        else:
            assert isinstance(dec_poolings, list), type(dec_poolings)
        self.dec_poolings = dec_poolings
        if dec_kernel_size is None:
            dec_kernel_size = kernel_size[:]
        else:
            if isinstance(dec_kernel_size, int):
                dec_kernel_size = [dec_kernel_size] * len(dec_fmaps)
        assert isinstance(dec_kernel_size, list), type(dec_kernel_size)
        # Build the decoder
        self.dec_blocks = nn.ModuleList()
        for pi, (fmap, pool, kw) in enumerate(zip(dec_fmaps, dec_poolings,
                                                  dec_kernel_size),
                                              start=1):
            if skip and pi > 1 and pool > 1:
                if skip_merge == 'concat':
                    ninp *= 2

            if pi >= len(dec_fmaps):
                act = 'Tanh'
            else:
                act = None
            if pool > 1:
                dec_block = GDeconv1DBlock(
                    ninp, fmap, kw, stride=pool,
                    norm_type=norm_type, bias=bias,
                    act=act
                )
            else:
                dec_block = GConv1DBlock(
                    ninp, fmap, kw, stride=1,
                    bias=bias,
                    norm_type=norm_type
                )
            self.dec_blocks.append(dec_block)
            ninp = fmap

        self.init_weights()
    def forward(self, x, z=None, ret_hid=False):
        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.enc_blocks):
            hi, linear_hi = enc_layer(hi, True)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
            #print('Adding skip[{}]={}, alpha={}'.format(l_i,
            #                                            hi.size(),
            #                                            hi.size(1)))
            if self.skip and l_i < (len(self.enc_blocks) - 1):
                skips[l_i]['tensor'] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        if not self.no_z:
            if z is None:
                # make z
                z = torch.randn(hi.size(0), self.z_dim, *hi.size()[2:])
                if hi.is_cuda:
                    z = z.to('cuda')
            if len(z.size()) != len(hi.size()):
                raise ValueError('len(z.size) {} != len(hi.size) {}'
                                 ''.format(len(z.size()), len(hi.size())))
            if not hasattr(self, 'z'):
                self.z = z
            hi = torch.cat((z, hi), dim=1)
            if ret_hid:
                hall['enc_zc'] = hi
        else:
            z = None
        enc_layer_idx = len(self.enc_blocks) - 1
        for l_i, dec_layer in enumerate(self.dec_blocks):
            if self.skip and enc_layer_idx in self.skips and \
                    self.dec_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                hi = skip_conn['alpha'](skip_conn['tensor'], hi)
            hi = dec_layer(hi)
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if ret_hid:
            return hi, hall
        else:
            return hi

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)


class Discriminator(nn.Module):

    def __init__(self, input_size,
                 kernel_size,
                 pool_type='none',
                 pool_slen= 16, # Dimension of last conv D layer time axis
                 norm_type='bnorm',
                 bias=True,
                 phase_shift=5,
                 sinc_conv=False,
                 name='Discriminator'):
        super().__init__()
        # phase_shift randomly occurs within D layers
        # as proposed in https://arxiv.org/pdf/1802.04208.pdf
        # phase shift has to be specified as an integer
        self.phase_shift = phase_shift
        self.poolings = [4]*5
        self.fmaps = [64, 128, 256, 512, 1024]
        self.name = name
        if phase_shift is not None:
            assert isinstance(phase_shift, int), type(phase_shift)
            assert phase_shift > 1, phase_shift
        if pool_slen is None:
            raise ValueError('Please specify D network pool seq len '
                             '(pool_slen) in the end of the conv '
                             'stack: [inp_len // (total_pooling_factor)]')
        ninp = input_size
        # SincNet as proposed in
        # https://arxiv.org/abs/1808.00158
        if sinc_conv:
            # build sincnet module as first layer
            self.sinc_conv = SincConv(self.fmaps[0] // 2,
                                      251, 16e3, padding='SAME')
            inp = self.fmaps[0]
            self.fmaps = self.fmaps[1:]
        self.enc_blocks = nn.ModuleList()
        for pi, (fmap, pool) in enumerate(zip(self.fmaps,
                                              self.poolings),
                                          start=1):
            enc_block = GConv1DBlock(
                ninp, fmap, kernel_size, stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_slen *= self.fmaps[-1]
            self.fc = nn.Sequential(
                nn.Linear(pool_slen, 256),
                nn.PReLU(256),
                nn.Linear(256, 128),
                nn.PReLU(128),
                nn.Linear(128, 1)
            )
            if norm_type == 'snorm':
                spectral_norm(self.fc[0])
                spectral_norm(self.fc[2])
                spectral_norm(self.fc[3])
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(self.fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_slen, 1)
            if norm_type == 'snorm':
                spectral_norm(self.pool_conv)
                spectral_norm(self.fc)
        elif pool_type == 'gmax':
            self.gmax = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(self.fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gavg':
            self.gavg = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(self.fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Conv1d(self.fmaps[-1], self.fmaps[-1], 1),
                nn.PReLU(self.fmaps[-1]),
                nn.Conv1d(self.fmaps[-1], 1, 1)
            )
            if norm_type == 'snorm':
                spectral_norm(self.mlp[0])
                spectral_norm(self.mlp[1])
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)
        self.init_weights()
    def forward(self, x):
        h = x
        if hasattr(self, 'sinc_conv'):
            h_l, h_r = torch.chunk(h, 2, dim=1)
            h_l = self.sinc_conv(h_l)
            h_r = self.sinc_conv(h_r)
            h = torch.cat((h_l, h_r), dim=1)
        # store intermediate activations
        int_act = {}
        for ii, layer in enumerate(self.enc_blocks):
            if self.phase_shift is not None:
                shift = random.randint(1, self.phase_shift)
                # 0.5 chance of shifting right or left
                right = random.random() > 0.5
                # split tensor in time dim (dim 2)
                if right:
                    sp1 = h[:, :, :-shift]
                    sp2 = h[:, :, -shift:]
                    h = torch.cat((sp2, sp1), dim=2)
                else:
                    sp1 = h[:, :, :shift]
                    sp2 = h[:, :, shift:]
                    h = torch.cat((sp2, sp1), dim=2)
            h = layer(h)
            int_act['h_{}'.format(ii)] = h
        if self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
            y = self.fc(h)
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gmax':
            h = self.gmax(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gavg':
            h = self.gavg(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'mlp':
            y = self.mlp(h)
        int_act['logit'] = y
        return y, int_act

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

def print_summary(print_disc=True, print_gen=True, kernel_size=31, stride=2):
    from torchsummary import summary
    if print_gen:
        G = Generator(1,
                      kernel_size=31,
                       no_z=True)
        # z = nn.init.normal_(torch.Tensor(1, 1024, 7))
        # z = torch.randn([1] + G.get_z_shape())
        z = torch.randn([1] + [1024, 8])
        x = torch.randn(1, 1, 16384)
        print("Generator Network")
        summary(G, [x.shape[1:], z.shape[1:]])
    if print_disc:
        D = Discriminator(2, 31, pool_type='none',
                             pool_slen=16)
        x = torch.randn(1, 2, 16384)
        print("Discriminator Network")
        # summary(D, x)
        summary(D, x.shape[1:])


# print_summary()
