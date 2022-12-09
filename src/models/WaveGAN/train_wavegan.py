from torch import autograd
from torch import optim
from icecream import ic
import os
import datetime
from torch.utils.data import DistributedSampler, DataLoader
from datasets import WaveGANDataset, split_data, get_npy_filename_list
import yaml
from wavegan import *
import time
from utils import (config_from_yaml, print_training_stats,
                   scan_checkpoint, load_checkpoint, save_checkpoint,
                   plot_generated_samples)
import click

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@click.command()
# options_metavar='<options>'
@click.option('--epochs', default=100, type=int,
              help='Number of epochs to train the GAN for')
@click.option('--summary_interval', default=500, type=int,
              help='Iteration steps at which to plot FRFs')
@click.option('--save_interval', default=1000, type=int,
              help='Iteration steps at which to save checkpoints')
@click.option('--validate', is_flag=True,
              help='Use validation data to confirm model has not overfit')
@click.option('--use_wandb', is_flag=True,
              help='Use weights and biases to monitor training')
@click.option('--config_file', default='HiFiGAN_config.yaml', type=str,
              help='Configuration (.yaml) file including hyperparameters for training')
def run_wavegan(epochs,
                summary_interval,
                save_interval,
                validate,
                config_file,
                use_wandb):
    if use_wandb:
        import wandb
        date_ = datetime.date.today().strftime('%m-%d')
        time_ = datetime.datetime.now().strftime("%H:%M")
        print("Using Weights and Biases to track training!")
        wandb.login()
        config_dict = yaml.load(open(config_file), Loader=yaml.FullLoader)
        ic(config_dict)
        run = wandb.init(project='WaveGAN_training',
                         name=config_dict['model_name']['value'] + '_' + date_ + '_' + time_,
                         config=config_file)
        config = wandb.config
    else:
        config = config_from_yaml(config_file)
    # init params
    config.epochs = epochs

    # =============Network===============
    netG = WaveGANGenerator(model_size=config.model_size, ngpus=1, latent_dim=config.latent_dim, upsample=True)
    netD = WaveGANDiscriminator(model_size=config.model_size, ngpus=1)

    # =========Checkpoint Init===========
    # define checkpoint directory
    checkpoint_dir = "Checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    chkpt_dir = os.path.join(checkpoint_dir, config.model_name)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    # look for previous training checkpoints to load
    if os.path.isdir(chkpt_dir):
        cp_g = scan_checkpoint(chkpt_dir, 'g_')
        cp_d = scan_checkpoint(chkpt_dir, 'd_')
    # initialise steps and restore checkpoints (if they exist)
    steps = 0
    if cp_g is None or cp_d is None:
        state_dict_d = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_d = load_checkpoint(cp_d, device)
        netG.load_state_dict(state_dict_g['netG'])
        netD.load_state_dict(state_dict_d['netD'])
        steps = state_dict_d['steps'] + 1
        last_epoch = state_dict_d['epoch']

    # ============DataSet==============
    dset_filenames = get_npy_filename_list(config.data_directory)
    train_files, valid_files, _, train_size = split_data(dset_filenames, 0.05, 0.)

    train_dset = WaveGANDataset(train_files, config.rir_length, sampling_rate=config.sample_rate)
    # data loader function
    train_loader = DataLoader(train_dset, num_workers=config.num_workers, shuffle=True,
                              sampler=None,
                              batch_size=config.batch_size,
                              pin_memory=True,
                              drop_last=True)
    total_batches = len(train_loader)
    if validate:
        valid_dset = WaveGANDataset(valid_files, config.rir_length, sampling_rate=config.sample_rate)

        valid_loader = DataLoader(valid_dset, num_workers=config.num_workers, shuffle=True,
                                  sampler=None,
                                  batch_size=config.batch_size,
                                  pin_memory=True,
                                  drop_last=True)

    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()

    # "Two time-scale update rule"(TTUR) to update netD 4x faster than netG.
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    if state_dict_d is not None:
        optimizerG.load_state_dict(state_dict_d['optimizerG'])
        optimizerD.load_state_dict(state_dict_d['optimizerD'])

    # Sample noise used for generated output.
    sample_noise = torch.randn(config.sample_size, config.latent_dim)
    if cuda:
        sample_noise = sample_noise.cuda()
    static_sample_latent = autograd.Variable(sample_noise, requires_grad=False)


    # =============Init_logs===============
    outputs = {}
    outputs['D_costs_train'] = []
    outputs['D_wasses_train'] = []
    if validate:
        outputs['D_costs_valid'] = []
        outputs['D_wasses_valid'] = []
    outputs['G_costs'] = []

    start = time.time()
    # =============Train===============
    for epoch in range(max(0, last_epoch), epochs):
        epoch_metrics = {}
        epoch_metrics['D_cost_train_epoch'] = []
        epoch_metrics['D_wass_train_epoch'] = []
        if validate:
            epoch_metrics['D_cost_valid_epoch'] = []
            epoch_metrics['D_wass_valid_epoch'] = []
        epoch_metrics['G_cost_epoch'] = []

        for i, data in enumerate(train_loader):
            rirs, filepath = data
            # for i in range(1, BATCH_NUM+1):
            # Set Discriminator parameters to require gradients.

            for p in netD.parameters():
                p.requires_grad = True

            one = torch.tensor(1, dtype=torch.float)
            neg_one = one * -1
            if cuda:
                one = one.cuda()
                neg_one = neg_one.cuda()
            #############################
            # (1) Train Discriminator
            #############################
            for iter_dis in range(config.n_D_updates):
                netD.zero_grad()

                # Noise
                noise = torch.Tensor(config.batch_size, config.latent_dim).normal_(0., 1.)
                if cuda:
                    noise = noise.cuda()
                    rirs = rirs.cuda()
                noise_Var = torch.autograd.Variable(noise, requires_grad=False)

                # a) compute loss contribution from real training data
                D_real = netD(rirs)
                D_real = D_real.mean()  # avg loss
                D_real.backward(neg_one)  # loss * -1

                # b) compute loss contribution from generated data, then backprop.
                fake = autograd.Variable(netG(noise_Var).data)
                D_fake = netD(fake)
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # c) compute gradient penalty and backprop
                gradient_penalty = calc_gradient_penalty(netD, rirs.data,
                                                         fake.data, config.batch_size, config.gradient_penalty_weight,
                                                         use_cuda=cuda)
                gradient_penalty.backward(one)

                # Compute cost * Wassertein loss..
                D_cost_train = D_fake - D_real + gradient_penalty
                D_wass_train = D_real - D_fake

                # Update gradient of discriminator.
                optimizerD.step()

                #############################
                # (2) Compute Valid data
                #############################
                netD.zero_grad()
                if validate:
                    valid_data = next(valid_loader)
                    if cuda:
                        valid_data = valid_data.cuda()
                    D_real_valid = netD(valid_data)
                    D_real_valid = D_real_valid.mean()  # avg loss

                    # b) compute loss contribution from generated data, then backprop.
                    fake_valid = netG(noise_Var)
                    D_fake_valid = netD(fake_valid)
                    D_fake_valid = D_fake_valid.mean()

                    # c) compute gradient penalty and backprop
                    gradient_penalty_valid = calc_gradient_penalty(netD, valid_data.data,
                                                                   fake_valid.data, config.batch_size,
                                                                   config.gradient_penalty_weight,
                                                                   use_cuda=cuda)
                    # Compute metrics and record in batch history.
                    if validate:
                        D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
                        D_wass_valid = D_real_valid - D_fake_valid

                if cuda:
                    D_cost_train = D_cost_train.cpu()
                    D_wass_train = D_wass_train.cpu()
                    if validate:
                        D_cost_valid = D_cost_valid.cpu()
                        D_wass_valid = D_wass_valid.cpu()

                # Record costs
                epoch_metrics['D_cost_train_epoch'].append(D_cost_train.data.numpy())
                epoch_metrics['D_wass_train_epoch'].append(D_wass_train.data.numpy())
                if validate:
                    epoch_metrics['D_cost_valid_epoch'].append(D_cost_valid.data.numpy())
                    epoch_metrics['D_wass_valid_epoch'].append(D_wass_valid.data.numpy())

            #############################
            # (3) Train Generator
            #############################
            # Prevent discriminator update.
            for p in netD.parameters():
                p.requires_grad = False

            # Reset generator gradients
            netG.zero_grad()

            # Noise
            noise = torch.Tensor(config.batch_size, config.latent_dim).normal_(0., 1.)
            if cuda:
                noise = noise.cuda()
            noise_Var = torch.autograd.Variable(noise, requires_grad=False)

            fake = netG(noise_Var)
            G = netD(fake)
            G = G.mean()

            # Update gradients.
            G.backward(neg_one)
            G_cost = -G

            optimizerG.step()

            # Record costs
            if cuda:
                G_cost = G_cost.cpu()
            epoch_metrics['G_cost_epoch'].append(G_cost.data.numpy())
            # print train info
            print_training_stats(epoch, epochs, total_batches, epoch_metrics, i, start_time = start)

            # checkpointing
            if steps % save_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(chkpt_dir, steps)
                save_checkpoint(chkpt_dir,
                                checkpoint_path,
                                {'netG': netG.state_dict()},
                                remove_below_step= steps // 5)
                checkpoint_path = "{}/d_{:08d}".format(chkpt_dir, steps)
                save_checkpoint(chkpt_dir,
                                checkpoint_path,
                                {'netD': netD.state_dict(),
                                 'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict(),
                                 'steps': steps,
                                 'epoch': epoch},
                                remove_below_step= steps // 5)


            if steps % summary_interval == 0:
                temp_dict = {'PerEpoch/steps': steps}
                new_dict = {}
                for key in epoch_metrics.keys():
                    new_dict['PerEpoch/' + key] = epoch_metrics[key][-1]
                log_dict = dict(new_dict, **temp_dict)
                if use_wandb:
                    wandb.log(log_dict)
                # sample and plot rirs
                sample_out = netG(static_sample_latent)
                if cuda:
                    sample_out = sample_out.cpu()
                sample_out = sample_out.squeeze(1).data.numpy()
                fig = plot_generated_samples(sample_out, fs = config.sample_rate)
                if use_wandb:
                    wandb.log({"Samples generated": fig})
            steps += 1

        # Save the average cost of batches in every epoch.
        D_cost_train_epoch_avg = sum(epoch_metrics['D_cost_train_epoch']) / float(len(epoch_metrics['D_cost_train_epoch']))
        D_wass_train_epoch_avg = sum(epoch_metrics['D_wass_train_epoch']) / float(len(epoch_metrics['D_wass_train_epoch']))
        if validate:
            D_cost_valid_epoch_avg = sum(epoch_metrics['D_cost_valid_epoch']) / float(len(epoch_metrics['D_cost_valid_epoch']))
            D_wass_valid_epoch_avg = sum(epoch_metrics['D_wass_valid_epoch']) / float(len(epoch_metrics['D_wass_valid_epoch']))
        G_cost_epoch_avg = sum(epoch_metrics['G_cost_epoch']) / float(len(epoch_metrics['G_cost_epoch']))

        outputs['D_costs_train'].append(D_cost_train_epoch_avg)
        outputs['D_wasses_train'].append(D_wass_train_epoch_avg)
        if validate:
            outputs['D_costs_valid'].append(D_cost_valid_epoch_avg)
            outputs['D_wasses_valid'].append(D_wass_valid_epoch_avg)
        outputs['G_costs'].append(G_cost_epoch_avg)


        temp_dict = {'Total/steps': steps, 'Total/epoch' : epoch}
        new_dict = {}
        for key in outputs.keys():

            if len(outputs[key]) > 1:
                new_dict['Total/' + key] = outputs[key][-1]
            else:
                new_dict['Total/' + key] = outputs[key][0]
        log_dict = dict(new_dict, **temp_dict)
        if use_wandb:
            wandb.log(log_dict)

if __name__ == '__main__':
    run_wavegan()
