import torch
import numpy as np
import sys
# sys.path.insert(0, "../../")
from src.models.WaveGAN.wavegan import *
from src.models.CSGM.utils import (config_from_yaml, print_training_stats,
                   scan_checkpoint, load_checkpoint, save_checkpoint,
                   plot_generated_samples, normalize)
from tqdm.auto import tqdm

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TotalVariation(nn.Module):
    def __init__(self, TV_weight=1., ndim=2):
        super(TotalVariation, self).__init__()
        self.TV_weight = TV_weight
        self.ndims = ndim

    def forward(self, x):
        if self.ndims > 2:
            h_x = x.size()[1]
            w_x = x.size()[2]
            count_h = self._tensor_size(x[:, 1:, :])
            count_w = self._tensor_size(x[:, :, 1:])
            h_tv = torch.pow(torch.abs(x[:, 1:, :] - x[:, :h_x - 1, :]), 2).sum()
            w_tv = torch.pow(torch.abs(x[:, :, 1:] - x[:, :, :w_x - 1]), 2).sum()
            return self.TV_weight * torch.sqrt(h_tv / count_h + w_tv / count_w)
        else:
            w_x = x.size()[1]
            count_w = self._tensor_size(x[:, 1:])
            w_tv = torch.abs(x[:, 1:] - x[:, :w_x - 1]).sum()
            return self.TV_weight * (w_tv / count_w)

    def _tensor_size(self, t):
        if self.ndims > 2:
            return t.size()[0] * t.size()[1] * t.size()[2]
        else:
            return t.size()[0] * t.size()[1]


def mask_frequencies(x, index):
    mask = torch.zeros_like(x)
    if mask.ndim > 2:
        mask[:, :index, :] = 1. + 1j * 0.
    else:
        mask[:, :index] = 1. + 1j * 0.
    return mask * x

def elastic_net(x, config, l1_weight = 0.1, l2_weight = 0.1):
    x = x.squeeze(0)
    # early_time = int(config.sample_rate*0.03)
    # l1norm = torch.linalg.norm(x[..., :early_time + 200], ord = 1, keepdim=True)
    # l2norm = torch.linalg.norm(x[..., early_time - 200:], ord = 2, keepdim=True)
    early_time = int(config.sample_rate*0.03)
    maxind = torch.argmax(x, dim = -1)
    t = torch.arange(0,x.shape[-1]).to(device)
    y = torch.exp((-t + early_time)/maxind)
    y1 = torch.clamp(y, min = 0, max = 1.).to(device)
    y2 = torch.erf(torch.linspace(0,10,x.shape[-1] )).to(device)
    l1norm = torch.linalg.norm(y1*x, ord = 1, keepdim=True)
    l2norm = torch.linalg.norm(y2*x, ord = 2, keepdim=True)

    return l1_weight*l1norm + l2_weight*l2norm

def regularisation_norm(x, Ax, config):
    if config.regularisor == 'masked':
        if config.reg_norm != 'tv':
            norm_ = torch.linalg.norm(Ax, ord=float(config.reg_norm), axis=(-2, -1), keepdim=True)
        else:
            if Ax.shape[1] == 1:
                Ax = Ax.squeeze(1)

            norm_ = TotalVariation(1., Ax.ndim)(Ax)
    else:
        if config.reg_norm != 'tv':
            norm_ = torch.linalg.norm(x, ord=float(config.reg_norm), axis=(-2, -1), keepdim=True)
        else:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            norm_ = TotalVariation(1., x.ndim)(x)
    return norm_.squeeze(0).squeeze(0).squeeze(0)


def load_generator(chkpt_dir, config):
    # =============Network===============
    G = WaveGANGenerator(model_size=config.model_size,
                         ngpus=1,
                         latent_dim=config.latent_dim,
                         upsample=True)
    cp_g = scan_checkpoint(chkpt_dir, 'g_')
    state_dict_g = load_checkpoint(cp_g, device)
    G.load_state_dict(state_dict_g['netG'])
    return G


def measurement_operator(x, config, use_stft=False, cut_off=900, return_same = False):

    if return_same:
        return x

    f = np.fft.rfftfreq(n=config.spec_nfft, d=1 / config.sample_rate)
    f_indx = np.argmin(f <= cut_off)

    if use_stft:
        while x.ndim > 2:
            x = x.squeeze(0)
        X = torch.stft(input=x,
                       n_fft=config.spec_nfft,
                       hop_length=config.spec_nfft // 4,
                       center=True,
                       onesided=True,
                       normalized = False,
                       return_complex=True)  # [..., freq, time]
    else:
        X = torch.fft.rfft(x, n=config.sample_size)

    Xm = mask_frequencies(X, f_indx)

    return Xm


def CSGM(y, G, config, run):
    # initiate
    G.eval()
    G.to(device)
    y = y.to(device)
    objective = lambda target, prediction: (torch.abs(target - prediction) ** 2).mean()
    # objective = RandomResolutionSTFTLoss(resolutions=3, max_fft_size=4096)
    z_init = torch.normal(torch.zeros(config.n_z_init, config.latent_dim)).to(device)

    csgm_losses = torch.zeros(config.n_z_init).to(device)
    # The following loop can be replaced by optimization in parallel, here it used
    # due to memory limitation
    Y = measurement_operator(y, config, use_stft=config.use_stft, cut_off=config.cut_off) # Ax
    losses = {}
    pbar1 = tqdm(range(config.n_z_init))
    print('Running CSGM:')

    for i in pbar1:
        # if(config.n_z_init > 1):
        #     print('Z initialization number %d/%d' %(i+1, config.n_z_init))
        pbar1.set_description('Z initialization number %d/%d' % (i + 1, config.n_z_init))
        Z = torch.autograd.Variable(z_init[i:i + 1, :], requires_grad=True)
        optimizer = torch.optim.Adam([{'params': Z, 'lr': config.lr_z_CSGM}])

        losses[f'objective_{i}'] = []
        losses[f'regularisation_{i}'] = []
        losses[f'total_{i}'] = []

        pbar2 = tqdm(range(config.CSGM_iterations), leave=False)
        for step in pbar2:
            optimizer.zero_grad()
            # Gz = normalize(G(Z))*0.95
            Gz = G(Z)
            AGz = measurement_operator(Gz, config, use_stft=config.use_stft, cut_off=config.cut_off)
            loss = objective(AGz, Y)
            losses[f'objective_{i}'].append(float(loss.cpu().detach().numpy()))
            # reg = config.reg_weight * regularisation_norm(Gz, AGz, config)
            # reg = config.reg_weight * torch.linalg.norm(Z, ord = 2).squeeze(0)
            reg = elastic_net(Gz.squeeze(0), config, config.el_net_l1_weight, config.el_net_l2_weight)
            losses[f'regularisation_{i}'].append(float(reg.cpu().detach().numpy()))
            total_loss = loss + reg
            losses[f'total_{i}'].append(float(total_loss.cpu().detach().numpy()))

            if run is not None:
                run.log({f'CSGM_objective_{i}': loss.cpu().detach().item(), 'steps': step})
                run.log({f'CSGM_regularisation_{i}': reg.cpu().detach().item(), 'steps': step})
                run.log({f'CSGM_total_{i}': total_loss.cpu().detach().item(), 'steps': step})

            loss.backward()
            optimizer.step()
            # if(step % config.print_every == 0):
            #     print('CSGM step %d/%d, objective = %.5f' %(step, config.CSGM_iterations, loss.item()))
            pbar2.set_description(
                'CSGM step %d/%d, objective = %.5f' % (step + 1, config.CSGM_iterations, total_loss.item()))
        csgm_losses[i] = objective(AGz, Y)
        max_indx = torch.argmax(Gz, dim = -1)
        if max_indx > int(0.100*16000):
            csgm_losses[i] = 10000
        z_init[i:i + 1, :] = Z.detach()
    z_hat_idx = torch.argmin(csgm_losses)
    z_hat = z_init[z_hat_idx:z_hat_idx + 1, :]
    I_CSGM = G(z_hat)

    return I_CSGM, z_hat, losses


def AdaptiveCSGM(y, G, z_hat, config, run):
    # initiate
    G.eval()
    G.to(device)
    Z = torch.autograd.Variable(z_hat, requires_grad=True).to(device)
    y = y.to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=config.lr_G)
    optimizer_z = torch.optim.Adam([{'params': Z, 'lr': config.lr_z_CSGM}])
    objective = lambda target, prediction: (torch.abs(target - prediction) ** 2).mean()
    # objective = RandomResolutionSTFTLoss(resolutions=3, max_fft_size=4096)
    Y = measurement_operator(y, config, use_stft=config.use_stft, cut_off=config.cut_off)
    print('Running adaptive stage:')
    losses = {}
    losses['objective'] = []
    losses['regularisation'] = []
    losses['total'] = []
    pbar1 = tqdm(range(config.adaptive_iters), leave=True)
    for step in pbar1:
        optimizer_z.zero_grad()
        optimizer_G.zero_grad()
        Gz = G(Z)
        AGz = measurement_operator(Gz, config, use_stft=config.use_stft, cut_off=config.cut_off)
        loss = objective(AGz, Y)
        losses['objective'].append(float(loss.cpu().detach().numpy()))
        # reg = config.ad_reg_weight * regularisation_norm(Gz, AGz, config)
        reg = elastic_net(Gz, config, config.el_net_l1_weight, config.el_net_l2_weight)
        losses['regularisation'].append(float(reg.cpu().detach().numpy()))
        total_loss = loss + reg
        losses['total'].append(float(total_loss.cpu().detach().numpy()))

        if run is not None:
            run.log({'adapt_CSGM_objective': loss.cpu().detach().item(), 'ad_steps': step})
            run.log({'adapt_CSGM_regularisation': reg.cpu().detach().item(), 'ad_steps': step})
            run.log({'adapt_CSGM_total': total_loss.cpu().detach().item(), 'ad_steps': step})

        total_loss.backward()
        optimizer_G.step()
        optimizer_z.step()
        # if(step % config.print_every == 0):
        #     print('IA step %d/%d, objective = %.5f' %(step, config.adaptive_iters, loss.item()))
        pbar1.set_description(
            'Adaptive GAN step %d/%d, objective = %.5f' % (step, config.adaptive_iters, total_loss.item()))
    return Gz.detach(), losses
