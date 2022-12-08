import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm
from icecream import ic


class NNconv(nn.Module):
    """ Nearest neighbour interpolation"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=31,
                 upsample_factor=2,
                 padding=0,
                 mode='nearest',
                 use_spectral_norm=True
                 ):
        super(NNconv, self).__init__()
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_spectral_norm = use_spectral_norm
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode=mode)
        if use_spectral_norm:
            self.conv1d = spectral_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=padding))
        else:
            self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=1, padding=padding)

        nn.init.xavier_normal_(self.conv1d.weight.data)

    def forward(self, x):

        x_up = self.upsample(x)
        x_out = self.conv1d(x_up)
        return x_out

class SubPixelConv(nn.Module):
    """ Sub pixel Convolution"""

    def __init__(self,
                 in_channels,
                 out_feature_len,
                 kernel_size=31,
                 upsample_factor=2,
                 padding=0,
                 mode='nearest',
                 use_spectral_norm=True
                 ):
        super(SubPixelConv, self).__init__()
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_spectral_norm = use_spectral_norm
        self.out_feature_len = out_feature_len
        if use_spectral_norm:
            self.conv1d = spectral_norm(nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                                  kernel_size=kernel_size, stride=1, padding=padding))
        else:
            self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, stride=1, padding=padding)

        nn.init.xavier_normal_(self.conv1d.weight.data)

    def shuffle(self, x):
        batch_size, channels, steps = x.size()
        channels //= self.upsample_factor
        input_view = x.contiguous().view(batch_size, channels, self.upsample_factor, steps)
        shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
        return shuffle_out.view(batch_size, channels, steps * self.upsample_factor)

    def forward(self, x):

        x_up = self.conv1d(x)
        x_out = self.shuffle(x_up)
        if x_out.shape[-1] != self.out_feature_len:
            x_out = nn.functional.pad(x_out, (0, self.out_feature_len - x_out.shape[-1]))

        return x_out

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 8)
        self.key_conv = nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 8)
        self.value_conv = nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X S)
            returns :
                out : self attention value + input feature
                attention: B X S X N (N is Width*Height)
        """
        m_batchsize, C, samples = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, samples).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, samples)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, samples)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, samples)

        out = self.gamma * out + x
        return out, attention

class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.
    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # define gamma and beta parameters
        self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        Args:
            x: input tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        Result:
            x: normalized batch tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features
            mean_sq: squared means over features
        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


class Generator(nn.Module):
    """G"""

    def __init__(self, kernel_size=32, upsamp_ratio=2, interp_conv=True, use_attention = False):
        super().__init__()
        # encoder gets a noisy signal as input [B x 1 x 16384]
        self.upsamp_ratio = upsamp_ratio
        self.input_length = 16384
        self.g_enc_depths = [16, 32, 32, 64, 64, 128]
        self.g_dec_depths = [128, 128, 64, 64, 32, None]
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        # self.padding = kernel_size // 2 - 1
        self.padding = 15
        encvarname = 'enc'
        decvarname = 'dec'
        if not interp_conv:
            transpconv = lambda a, b, c, d, e: spectral_norm(nn.ConvTranspose1d(a, b, c, d, e))
        else:
            transpconv = lambda a, b, c, d, e: NNconv(a, b, 31, d, e + 1)
            # transpconv = lambda a, b, c, d, e: SubPixelConv(a, b, 31, d, e + 1)

        for ii, d in enumerate(self.g_enc_depths):
            if ii == 0:
                setattr(self, encvarname + f'{ii + 1}', spectral_norm(nn.Conv1d(1, d,
                                                                                self.kernel_size,
                                                                                self.upsamp_ratio,
                                                                                self.padding)))
                setattr(self, encvarname + f'{ii + 1}_nl', nn.PReLU())
            else:
                setattr(self, encvarname + f'{ii + 1}',
                        spectral_norm(nn.Conv1d(self.g_enc_depths[ii - 1], d,
                                                self.kernel_size + 1,
                                                self.upsamp_ratio,
                                                self.padding)))
                setattr(self, encvarname + f'{ii + 1}_nl', nn.PReLU())

        for ii, d in enumerate(self.g_dec_depths):
            if ii == 0:
                setattr(self, decvarname + f'{len(self.g_enc_depths) - ii - 1}',
                        transpconv(256,
                                   64,
                                   self.kernel_size + 1,
                                   self.upsamp_ratio,
                                   self.padding - 1))
                setattr(self, decvarname + f'{len(self.g_enc_depths) - ii - 1}_nl_attention',
                        Self_Attn(256 // 2))

                setattr(self, decvarname + f'{len(self.g_enc_depths) - ii - 1}_nl', nn.PReLU())
            elif ii < len(self.g_dec_depths) - 1:
                setattr(self, decvarname + f'{len(self.g_enc_depths) - ii - 1}',
                        transpconv(self.g_dec_depths[ii - 1],
                                   d // 2, self.kernel_size + 1, self.upsamp_ratio,
                                   self.padding - 1))
                setattr(self, decvarname + f'{len(self.g_enc_depths) - ii - 1}_nl_attention',
                        Self_Attn(self.g_dec_depths[ii]))

                setattr(self, decvarname + f'{len(self.g_enc_depths) - ii - 1}_nl', nn.PReLU())
            else:
                setattr(self, 'dec_final', transpconv(32, 1,
                                                      self.kernel_size + 1,
                                                      self.upsamp_ratio,
                                                      self.padding - 1))
                setattr(self, 'dec_tanh', nn.Tanh())

        # decoder generates an enhanced signal
        # each decoder output are concatenated with homologous encoder output,
        # so the feature map sizes are doubled

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, z):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal)
            z: latent vector
        """
        # encoding step
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        # c = compressed feature, the 'thought vector'
        c = self.enc6_nl(e6)
        # concatenate the thought vector with latent variable
        encoded = torch.cat((c, z), dim=1)
        # decoding step
        d5 = self.dec5(encoded)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        if self.use_attention:
            d5_c, att_5 = self.dec5_nl_attention(d5_c)
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        if self.use_attention:
            d4_c, att_4 = self.dec4_nl_attention(d4_c)
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        if self.use_attention:
            d3_c, att_3 = self.dec3_nl_attention(d3_c)
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        # d2_c, att_2 = self.dec2_nl_attention(d2_c)
        d1 = self.dec1(d2_c)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = nn.functional.pad(d1, (0, e1.shape[-1] - d1.shape[-1]), mode="reflect")
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        # d1_c, att_1 = self.dec1_nl_attention(d1_c)
        out = self.dec_tanh(self.dec_final(d1_c))
        if out.shape[-1] < self.input_length:
            out = nn.functional.pad(out, (0, abs(self.input_length - out.shape[-1])), mode="reflect")
        elif out.shape[-1] > self.input_length:
            out = out[..., :self.input_length]
        return out

    def get_z_shape(self):
        if self.kernel_size == self.upsamp_ratio:
            return [self.g_enc_depths[-1], 13]
        else:
            if self.kernel_size / 2 - self.padding > 1.:
                remainder = 1
            else:
                remainder = 0
            return [self.g_enc_depths[-1],
                    self.input_length // (self.upsamp_ratio ** len(self.g_enc_depths)) - remainder]


class Discriminator(nn.Module):
    """D"""

    def __init__(self, kernel_size=31, downsample_ratio=2, use_attention = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        self.downsample_ratio = downsample_ratio
        # self.padding = kernel_size // 2 - 1
        self.padding = 15
        # D gets a noisy signal and clear signal as input [B x 2 x 16384]
        negative_slope = 0.03
        self.conv1 = spectral_norm(nn.Conv1d(in_channels=2, out_channels=16, kernel_size=self.kernel_size,
                                             stride=self.downsample_ratio,
                                             padding=self.padding))  # [B x 32 x 8192]
        self.self_att1 = Self_Attn(1)
        self.vbn1 = VirtualBatchNorm1d(16)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = spectral_norm(
            nn.Conv1d(16, 32, self.kernel_size, self.downsample_ratio, self.padding))  # [B x 64 x 4096]
        self.self_att2 = Self_Attn(32)
        self.vbn2 = VirtualBatchNorm1d(32)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = spectral_norm(
            nn.Conv1d(32, 32, self.kernel_size, self.downsample_ratio, self.padding))  # [B x 64 x 2048]
        self.self_att3 = Self_Attn(32)
        self.dropout1 = nn.Dropout()
        self.vbn3 = VirtualBatchNorm1d(32)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = spectral_norm(
            nn.Conv1d(32, 64, self.kernel_size, self.downsample_ratio, self.padding))  # [B x 128 x 1024]
        self.self_att4 = Self_Attn(64)
        self.vbn4 = VirtualBatchNorm1d(64)
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = spectral_norm(
            nn.Conv1d(64, 64, self.kernel_size, self.downsample_ratio, self.padding))  # [B x 128 x 512]
        self.self_att5 = Self_Attn(64)
        self.vbn5 = VirtualBatchNorm1d(64)
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = spectral_norm(
            nn.Conv1d(64, 128, self.kernel_size, self.downsample_ratio, self.padding))  # [B x 256 x 256]
        self.self_att6 = Self_Attn(128)
        self.dropout2 = nn.Dropout()
        self.vbn6 = VirtualBatchNorm1d(128)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = spectral_norm(
            nn.Conv1d(128, 256, self.kernel_size, self.downsample_ratio, self.padding))  # [B x 256 x 128]
        self.self_att7 = Self_Attn(256)
        self.vbn7 = VirtualBatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        # 1x1 size kernel for dimension and parameter reduction
        self.conv_final = spectral_norm(nn.Conv1d(128, 1, kernel_size=1, stride=1))  # [B x 1 x 8]
        self.lrelu_final = nn.LeakyReLU(negative_slope)

        if self.kernel_size == 4:
            in_feat = 13
        else:
            in_feat = 4
        self.fully_connected = nn.Linear(in_features=in_feat, out_features=1)  # [B x 1]
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, ref_x):
        """
        Forward pass of discriminator.
        Args:
            x: input batch (signal) - must be concatenated estimated and true signals
            ref_x: reference input batch for virtual batch norm
        """
        # reference pass
        ref_x = self.conv1(ref_x)
        ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
        ref_x = self.lrelu1(ref_x)
        ref_x = self.conv2(ref_x)
        ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
        ref_x = self.lrelu2(ref_x)
        ref_x = self.conv3(ref_x)
        ref_x = self.dropout1(ref_x)
        ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
        ref_x = self.lrelu3(ref_x)
        ref_x = self.conv4(ref_x)
        ref_x, mean4, meansq4 = self.vbn4(ref_x, None, None)
        ref_x = self.lrelu4(ref_x)
        ref_x = self.conv5(ref_x)
        ref_x, mean5, meansq5 = self.vbn5(ref_x, None, None)
        ref_x = self.lrelu5(ref_x)
        ref_x = self.conv6(ref_x)
        ref_x = self.dropout2(ref_x)
        ref_x, mean6, meansq6 = self.vbn6(ref_x, None, None)
        # further pass no longer needed

        # train pass
        x = self.conv1(x)
        x, _, _ = self.vbn1(x, mean1, meansq1)
        # x, att = self.self_att1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x, _, _ = self.vbn2(x, mean2, meansq2)
        x = self.lrelu2(x)
        if self.use_attention:
            x, att2 = self.self_att2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x, _, _ = self.vbn3(x, mean3, meansq3)
        x = self.lrelu3(x)
        if self.use_attention:
            x, att3 = self.self_att3(x)
        x = self.conv4(x)
        x, _, _ = self.vbn4(x, mean4, meansq4)
        x = self.lrelu4(x)
        if self.use_attention:
            x, att3 = self.self_att4(x)
        x = self.conv5(x)
        x, _, _ = self.vbn5(x, mean5, meansq5)
        x = self.lrelu5(x)
        if self.use_attention:
            x, att5 = self.self_att5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x, _, _ = self.vbn6(x, mean6, meansq6)
        x = self.lrelu6(x)
        if self.use_attention:
            x, att6 = self.self_att6(x)
        x = self.conv_final(x)

        x = self.lrelu_final(x)

        # reduce down to a scalar value
        x = torch.squeeze(x)

        x = self.fully_connected(x)
        return self.sigmoid(x)


def print_summary(print_disc=True, print_gen=True, kernel_size=31, stride=4):
    from torchsummary import summary
    if print_gen:
        G = Generator(kernel_size=kernel_size, upsamp_ratio=stride, interp_conv=False)
        # z = nn.init.normal_(torch.Tensor(1, 1024, 7))
        z = torch.randn([1] + G.get_z_shape())
        ic(G.get_z_shape())
        x = torch.randn(1, 1, 16384)
        print("Generator Network")
        summary(G, [x.shape[1:], z.shape[1:]])

    if print_disc:
        D = Discriminator(kernel_size=kernel_size, downsample_ratio=stride)
        x = torch.randn(1, 2, 16384)
        print("Discriminator Network")
        summary(D, [x.shape[1:], x.shape[1:]])


print_summary()
# x = torch.randn([2, 256, 13])
# subpix = SubPixelConv(256, 25, kernel_size = 31, upsample_factor = 4, padding = 15)
# out = subpix(x)
# out.shape


