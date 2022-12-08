import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn import CosineSimilarity
import torch.multiprocessing as mp
import gc

# from torch.distributed import init_process_group
from stft_loss import MultiResolutionSTFTLoss
from torch.nn.parallel import DistributedDataParallel
import sys

sys.path.insert(0, "../../")
from PlaneWaveData.TorchDataset import RIRDataset, ValidationDataset, find_files, spectrogram
from generator import Generator, generator_loss
from discriminator import (
    SpecDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    discriminator_loss,
)
from utils import (
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    plot_rir,
    plot_frf,
)
import click
import wandb
import matplotlib.pyplot as plt

from icecream import ic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(rank, train_epochs, train_dir, validation_dir, checkpoint_dir, hp):
    # if hp.train.num_gpus > 1:
    #     init_process_group(backend=hp.dist.dist_backend, init_method=hp.dist.dist_url,
    #                        world_size=hp.dist.world_size * hp.train.num_gpus, rank=rank)
    global start_b
    torch.cuda.manual_seed(hp.seed)
    torch.autograd.set_detect_anomaly(True)
    # call network architectures
    generator = Generator(hp.in_channels, hp.out_channels, num_layers= hp.G_layers,
                          num_stacks= hp.num_stacks, residual_channels= hp.residual_channels,
                          gate_channels= hp.gate_channels,
                          use_spectral_norm=hp.G_use_spectral_norm).to(device)
    specd = SpecDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    # stft loss for individual generator training
    stft_loss = MultiResolutionSTFTLoss(fft_sizes=hp.stft_loss_fft_sizes,
                                        hop_sizes = hp.stft_loss_hop_sizes,
                                        win_lengths = hp.stft_loss_win_lengths,
                                        window= hp.stft_loss_window)
    # use wandb to 'watch' network gradients
    wandb.watch(generator)
    wandb.watch(specd)
    wandb.watch(msd)
    # define checkpoint directory
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    chkpt_dir = os.path.join(checkpoint_dir, hp.chkpt_dir)
    if rank == 0:
        print("WaveNet Generator Summary: ")
        print("\n", generator)
        os.makedirs(chkpt_dir, exist_ok=True)
        print("checkpoints directory : ", chkpt_dir)

    # look for previous training checkpoints to load
    if os.path.isdir(chkpt_dir):
        cp_g = scan_checkpoint(chkpt_dir, "g_")
        cp_do = scan_checkpoint(chkpt_dir, "do_")

    # initialise steps and restore checkpoints (if they exist)
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        specd.load_state_dict(state_dict_do["specd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    # if there is over 1 gpu use distributed training strategy
    if hp.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        specd = DistributedDataParallel(specd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
    # set optimizer algorithms for G and D
    optim_g = torch.optim.AdamW(
        generator.parameters(), hp.G_lr, betas=[hp.G_opt_beta1, hp.G_opt_beta2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), specd.parameters()),
        hp.D_lr,
        betas=[hp.D_opt_beta1, hp.D_opt_beta2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.train.adam.lr_decay, last_epoch=last_epoch)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.train.adam.lr_decay, last_epoch=last_epoch)

    # ============DataSet==============
    if hp.use_ISM_data:
        query = "*.npz"
    else:
        query = "responses_sf_*.npz"
    dset_filenames = find_files(train_dir, query=query)
    # define trainset
    train_dset = RIRDataset(dset_filenames,
                            segment_size=hp.segment_length,
                            n_fft=hp.filter_length,
                            num_mels=hp.n_mel_channels,
                            hop_size=hp.hop_length,
                            win_size=hp.win_length,
                            sampling_rate=hp.sampling_rate,
                            fmin=hp.mel_fmin,
                            fmax=hp.mel_fmax,
                            n_cache_reuse=0,
                            shuffle=True,
                            device=device,
                            use_mel=hp.use_mel,
                            use_spectrogram=True)
    # for distributed training
    train_sampler = DistributedSampler(train_dset) if hp.num_gpus > 1 else None
    # data loader function
    train_loader = DataLoader(train_dset,
                              num_workers=hp.num_workers,
                              shuffle=True,
                              sampler=train_sampler,
                              batch_size=hp.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              )
    # total_batches = len(train_loader)

    valid_files = find_files(validation_dir, query="*.hdf5")
    validset = ValidationDataset(valid_files[0],
                                 segment_size=hp.segment_length,
                                 n_fft=hp.filter_length,
                                 num_mels=hp.n_mel_channels,
                                 hop_size=hp.hop_length,
                                 win_size=hp.win_length,
                                 sampling_rate=hp.sampling_rate,
                                 fmin=hp.mel_fmin,
                                 fmax=hp.mel_fmax,
                                 n_cache_reuse=0,
                                 device=device,
                                 use_mel=hp.use_mel,
                                 use_spectrogram=True)
    # data loader function
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=True,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )
    valid_iter = iter(validation_loader)

    # set networks as trainable
    generator.train()
    specd.train()
    msd.train()
    # use postnet?
    with_postnet = False
    for epoch in range(max(0, last_epoch), train_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if hp.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            if steps > hp.postnet_start_steps:
                with_postnet = False  # change if you want postnet in training
            x, y, file, y_spectro = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_spectro = y_spectro.to(device, non_blocking=True)
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

            y_generator, y_g_postnet = generator(
                x, with_postnet
            )  # no postnet output , postnet output
            # use postnet:
            if y_g_postnet is not None:
                y_g_spec = spectrogram(
                    y_g_postnet.squeeze(1),
                    hp.filter_length,
                    hp.n_mel_channels,
                    hp.sampling_rate,
                    hp.hop_length,
                    hp.win_length,
                    hp.mel_fmin,
                    hp.mel_fmax,
                    center=False,
                    use_mel=hp.use_mel,
                )
            # do not use postnet:
            else:
                y_g_spec = spectrogram(
                    y_generator.squeeze(1),
                    hp.filter_length,
                    hp.n_mel_channels,
                    hp.sampling_rate,
                    hp.hop_length,
                    hp.win_length,
                    hp.mel_fmin,
                    hp.mel_fmax,
                    center=False,
                    use_mel=hp.use_mel,
                )
            if steps > hp.discriminator_train_start_steps:
                for _ in range(hp.rep_discriminator):
                    optim_d.zero_grad()

                    # SpecD

                    y_df_hat_r, y_df_hat_g, _, _ = specd(y_spectro, y_g_spec.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                        y_df_hat_r, y_df_hat_g
                    )

                    # MSD
                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_generator.detach())
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                        y_ds_hat_r, y_ds_hat_g
                    )

                    loss_disc_all = loss_disc_s + loss_disc_f

                    loss_disc_all.backward()
                    optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            sc_loss, mag_loss = stft_loss(
                y_generator.reshape(y.shape).squeeze(1), y.squeeze(1)
            )
            before_loss_mel = sc_loss + mag_loss

            # L1 Sample Loss
            before_loss_sample = F.l1_loss(y, y_generator)
            loss_gen_all = hp.lambda_multiSTFT*before_loss_mel + hp.lambda_time_loss*before_loss_sample
            # pdb.set_trace()
            if y_g_postnet is not None:
                # L1 Mel-Spectrogram Loss
                # loss_mel = F.l1_loss(y_spectro_loss, y_g_spec)
                sc_loss_, mag_loss_ = stft_loss(
                    y_g_postnet[:, :, : y.size(2)].squeeze(1), y.squeeze(1)
                )
                loss_mel = sc_loss_ + mag_loss_
                # L1 Sample Loss
                loss_sample = F.l1_loss(y, y_g_postnet)
                loss_gen_all = loss_gen_all + (
                        hp.lambda_multiSTFT * loss_mel + hp.lambda_time_loss * loss_sample
                )

            if steps == hp.discriminator_train_start_steps:
                for g in optim_g.param_groups:
                    g["lr"] = 0.00001

            if steps > hp.discriminator_train_start_steps:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = specd(y_spectro, y_g_spec)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_generator)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_all + (
                        hp.lambda_advers_time * loss_gen_s
                        + hp.lambda_advers_TF * loss_gen_f
                        + hp.lambda_feat_match_time * loss_fm_s
                        + hp.lambda_feat_match_TF * loss_fm_f
                )

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % hp.summary_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_spectro, y_g_spec)
                        sample_error = F.l1_loss(y, y_generator)

                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Sample Error: {:4.3f}, "
                        "Mel-Spec. Error : {:4.3f}, time since last {} iters : {:4.3f}".format(
                            steps,
                            loss_gen_all.item(),
                            sample_error.item(),
                            mel_error.item(),
                            hp.summary_interval,
                            time.time() - start_b,
                        )
                    )

                # checkpointing
                if steps % hp.save_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(chkpt_dir, steps)
                    save_checkpoint(chkpt_dir,
                                    checkpoint_path,
                                    {
                                        "generator": (
                                            generator.module if hp.num_gpus > 1 else generator
                                        ).state_dict()
                                    },
                                    remove_below_step=steps // 3
                                    )
                    checkpoint_path = "{}/do_{:08d}".format(chkpt_dir, steps)
                    save_checkpoint(chkpt_dir,
                                    checkpoint_path,
                                    {
                                        "specd": (
                                            specd.module if hp.num_gpus > 1 else specd
                                        ).state_dict(),
                                        "msd": (
                                            msd.module if hp.num_gpus > 1 else msd
                                        ).state_dict(),
                                        "optim_g": optim_g.state_dict(),
                                        "optim_d": optim_d.state_dict(),
                                        "steps": steps,
                                        "epoch": epoch,
                                    },
                                    remove_below_step=steps // 3
                                    )

                # Wandb summary logging
                if steps % hp.summary_interval == 0:
                    # cos = CosineSimilarity(dim=-1, eps=1e-12)
                    # pearson = cos(
                    #     y - y.mean(dim=-1, keepdim=True),
                    #     y_generator - y_generator.mean(dim=-1, keepdim=True),
                    # )[0]

                    wandb.log(
                        {
                            "steps": steps,
                            "g_loss_train": loss_gen_all.item(),
                            "mel_spec_error": mel_error.item(),
                            "sample_error_(l1)": sample_error.item(),
                            # "pearson_corr_coeff": pearson.cpu().detach().item(),
                        }
                    )
                    if steps > hp.discriminator_train_start_steps:
                        wandb.log(
                            {
                                "steps": steps,
                                "Dspec_loss_real": torch.FloatTensor(losses_disc_f_r)
                                    .mean()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                "Dspec_loss_fake": torch.FloatTensor(losses_disc_f_g)
                                    .mean()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                            }
                        )
                        wandb.log(
                            {
                                "steps": steps,
                                "Dmulti_loss_real": torch.FloatTensor(losses_disc_s_r)
                                    .mean()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                "Dmulti_loss_fake": torch.FloatTensor(losses_disc_s_g)
                                    .mean()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                            }
                        )
                        wandb.log(
                            {"steps": steps, "D_loss_train_both": loss_disc_all.cpu().item()}
                        )

                        wandb.log(
                            {
                                "steps": steps,
                                "Gfeature_loss_spec": loss_fm_f.item(),
                                "Gfeature_loss_multi": loss_fm_s.item(),
                                "Gadv_loss_spec": loss_gen_f.item(),
                                "Gadv_loss_multi": loss_gen_s.item(),
                            }
                        )

                # Validation
                if steps % hp.validation_interval == 0:  # and steps != 0:
                    cos = CosineSimilarity(dim=-1, eps=1e-12)
                    ax1 = plot_frf(
                        y_truth=y[0, 0, :].cpu().numpy(),
                        y_pred=x[0, 0, :].cpu().numpy(),
                        title="train input responses",
                    )
                    ax2 = plot_frf(
                        y_truth=y[0, 0, :].cpu().numpy(),
                        y_pred=y_generator[0, 0, :].detach().cpu().numpy(),
                        title="train output responses",
                    )
                    wandb.log({"FRF_input_chart_train": wandb.Image(ax1)})
                    wandb.log({"FRF_output_chart_train": wandb.Image(ax2)})
                    plt.close("all")

                    ax3 = plot_rir(
                        y_truth=y[0, 0, :].cpu().numpy(),
                        y_pred=x[0, 0, :].cpu().numpy(),
                        title="train input rir",
                        y_pred_lab="Plane Wave Reconstruction RIR (Input)",
                    )
                    ax4 = plot_rir(
                        y_truth=y[0, 0, :].cpu().numpy(),
                        y_pred=y_generator[0, 0, :].detach().cpu().numpy(),
                        title="train output rir",
                        y_pred_lab="GAN Reconstruction RIR (Input)",
                    )
                    wandb.log({"RIR_eval_in_chart_train": wandb.Image(ax3)})
                    wandb.log({"RIR_eval_out_chart_train": wandb.Image(ax4)})
                    plt.close("all")

                    generator.eval()
                    val_err_tot = 0
                    pearson_r = 0
                    pearson_f = 0
                    with torch.no_grad():
                        for j in range(5):
                            batch = next(valid_iter)
                            x, y, y_spectro = batch
                            x = x.unsqueeze(1)
                            y = y.unsqueeze(1).to(device)
                            y_generator, y_g_postnet = generator(x.to(device))
                            if y_g_postnet is not None:
                                val_err_tot += F.l1_loss(y, y_g_postnet).item()
                            else:
                                val_err_tot += F.l1_loss(y, y_generator).item()

                            pearson_r += cos(y - y.mean(dim=-1, keepdim=True),
                                             x.to(device) - x.mean(dim=-1, keepdim=True).to(device)
                                             )[0].item()
                            pearson_f += cos(y - y.mean(dim=-1, keepdim=True),
                                             y_generator - y_generator.mean(dim=-1, keepdim=True)
                                             )[0].item()

                        ax1 = plot_frf(
                            y_truth=y[0, 0, :].cpu().numpy(),
                            y_pred=x[0, 0, :].cpu().numpy(),
                            title="validation input responses",
                        )
                        ax2 = plot_frf(
                            y_truth=y[0, 0, :].cpu().numpy(),
                            y_pred=y_generator[0, 0, :].detach().cpu().numpy(),
                            title="validation output responses",
                        )
                        wandb.log({"FRF_input_chart_validation": wandb.Image(ax1)})
                        wandb.log({"FRF_output_chart_validation": wandb.Image(ax2)})
                        plt.close("all")

                        ax3 = plot_rir(
                            y_truth=y[0, 0, :].cpu().numpy(),
                            y_pred=x[0, 0, :].cpu().numpy(),
                            title="validation input rir",
                            y_pred_lab="Plane Wave Reconstruction RIR (Input)",
                        )
                        ax4 = plot_rir(
                            y_truth=y[0, 0, :].cpu().numpy(),
                            y_pred=y_generator[0, 0, :].detach().cpu().numpy(),
                            title="validation output rir",
                            y_pred_lab="GAN Reconstruction RIR (Input)",
                        )
                        wandb.log({"RIR_eval_in_chart_valid": wandb.Image(ax3)})
                        wandb.log({"RIR_eval_out_chart_valid": wandb.Image(ax4)})
                        plt.close("all")

                        val_err = val_err_tot / (j + 1)
                        pearson_r_avg = pearson_r/(j + 1)
                        pearson_f_avg = pearson_f/(j + 1)
                        wandb.log({"Validation_error": val_err,
                                  "Pearson_input" : pearson_r_avg,
                                  "Pearson_output" : pearson_f_avg,
                                  "steps" : steps})

                        del val_err, x, y, y_generator
                    generator.train()
            if steps > hp.discriminator_train_start_steps:
                del (loss_gen_all, loss_fm_f, loss_fm_s, loss_gen_f, loss_disc_all,
                     loss_disc_f, loss_disc_s, loss_gen_s, losses_disc_f_g, losses_disc_s_g,
                     losses_disc_s_r, losses_disc_f_r, losses_gen_f, losses_gen_s, y_spectro, y_g_postnet,
                     y_df_hat_g, y_ds_hat_g, y_ds_hat_r, y_df_hat_r)
            else:
                del (loss_gen_all, before_loss_mel, before_loss_sample, sc_loss, mag_loss)
            gc.collect()
            steps += 1
            # print('\nsteps')
        # scheduler_g.step()
        # scheduler_d.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )


@click.command()
# options_metavar='<options>'
@click.option(
    "--train_dir", default="../PlaneWaveData/SoundFieldData", type=str, help="Directory of training data"
)
@click.option(
    "--validation_dir",
    default="../validation_responses",
    type=str,
    help="Directory of validation data",
)
@click.option(
    "--checkpoint_dir",
    default="./hifi_logD",
    type=str,
    help="Directory for saving model checkpoints",
)
@click.option(
    "--config_file",
    default="HiFiGAN_config.yaml",
    type=str,
    help="Hyper-parameter and network architecture details stored in a .yaml file",
)
@click.option(
    "--train_epochs",
    default=3100,
    type=int,
    help="Number of epochs for which to train HiFiGAN (in total)",
)
def HiFiTrain(train_dir, validation_dir, checkpoint_dir, config_file, train_epochs):
    print("Initializing Training Process on device: {}".format(device))

    wandb.init(config=config_file, allow_val_change=True, project="hifi_extension")

    torch.manual_seed(wandb.config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(wandb.config.seed)
        wandb.config.update({"num_gpus": torch.cuda.device_count()})
        wandb.config.batch_size = int(wandb.config.batch_size / wandb.config.num_gpus)
        print("Batch size per GPU :", wandb.config.batch_size)
    else:
        pass

    if wandb.config.num_gpus > 1:
        mp.spawn(train, nprocs=wandb.config.num_gpus, args=(wandb.config,))
    else:
        train(0, train_epochs, train_dir, validation_dir, checkpoint_dir, wandb.config)


if __name__ == "__main__":
    HiFiTrain()
