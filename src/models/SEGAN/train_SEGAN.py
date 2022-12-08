import warnings
import torch
from torch import optim, nn
from icecream import ic
import os
import datetime
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "../../")
from PlaneWaveData.TorchDataset import RIRDataset, ValidationDataset, find_files
import yaml
# from SEGAN_model import Generator, Discriminator
# from SEGAN_model_small import Generator, Discriminator
from SEGAN_plus import Generator, Discriminator
import time
from SEGAN_utils import (config_from_yaml, print_training_stats, plot_frf, plot_rir, save_checkpoint, load_checkpoint,
                         scan_checkpoint, generator_loss_fn)
import click
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


@click.command()
# options_metavar='<options>'
@click.option('--epochs', default=500, type=int,
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
def train(epochs,
          summary_interval,
          save_interval,
          validate,
          config_file,
          use_wandb):

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_wandb:
        import wandb
        date_ = datetime.date.today().strftime('%m-%d')
        time_ = datetime.datetime.now().strftime("%H:%M")
        print("Using Weights and Biases to track training!")
        wandb.login()
        config_dict = yaml.load(open(config_file), Loader=yaml.FullLoader)
        run = wandb.init(project='SEGAN_training',
                         name=config_dict['model_name']['value'] + '_' + date_ + '_' + time_,
                         config=config_file)
        config = wandb.config
    else:
        config = config_from_yaml(config_file)
    # init params
    config.epochs = epochs

    # =============Networks===============
    # discriminator = Discriminator(kernel_size= config.kernel_size, downsample_ratio= config.stride)
    # generator = Generator(kernel_size= config.kernel_size, upsamp_ratio= config.stride, interp_conv= config.interp_conv)
    discriminator = Discriminator(input_size = 2,kernel_size= config.kernel_size)
    generator = Generator(input_size = 1, kernel_size= config.kernel_size)

    # =========Checkpoint Init===========
    # define checkpoint directory
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    chkpt_dir = os.path.join(config.checkpoint_dir, config.model_name)
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
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_d['discriminator'])
        steps = state_dict_d['steps'] + 1
        last_epoch = state_dict_d['epoch']

    # ============DataSet==============
    query = "responses_sf_*.npz"
    dset_filenames = find_files(config.data_directory, query=query)
    # train_files, valid_files, _, train_size = split_data(dset_filenames, 0.2, 0.)

    train_dset = RIRDataset(dset_filenames, config.rir_length, config.nfft, sampling_rate=config.sample_rate)
    # reference set for Discriminators virtual batch norm
    reference_vbn_batch = train_dset.reference_batch(config.batch_size)

    reference_vbn_batch = torch.autograd.Variable(reference_vbn_batch.to(device))
    # data loader function
    train_loader = DataLoader(train_dset, num_workers=config.num_workers, shuffle=True,
                              sampler=None,
                              batch_size=config.batch_size,
                              pin_memory=True,
                              drop_last=True)
    total_batches = len(train_loader)
    if validate:
        valid_files = find_files(config.validation_dir, query="*.hdf5")
        valid_dset = ValidationDataset(valid_files[0], config.rir_length, config.nfft, sampling_rate=config.sample_rate)

        valid_loader = DataLoader(valid_dset, num_workers=config.num_workers, shuffle=True,
                                  sampler=None,
                                  batch_size= 1,
                                  pin_memory=True,
                                  drop_last=True,)
        valid_iter = iter(valid_loader)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # set optimizer algorithms for G and D
    if config.optimizer == 'adam':
        optimizerG = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        optimizerD = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    else:
        optimizerG = optim.RMSprop(generator.parameters(), lr=config.lr)
        optimizerD = optim.RMSprop(discriminator.parameters(), lr=config.lr)

    if state_dict_d is not None:
        optimizerG.load_state_dict(state_dict_d['optimizerG'])
        optimizerD.load_state_dict(state_dict_d['optimizerD'])

    # set networks as trainable
    generator.train()
    discriminator.train()
    # generator L1 loss

    gen_l1_loss = generator_loss_fn(config.loss_type)

    # =============Init_logs===============
    outputs_logs = {}
    outputs_logs['D_cost_fake'] = []
    outputs_logs['D_cost_real'] = []
    if validate:
        validation_logs = {}
        validation_logs['Validation/D_cost_fake_valid'] = []
        validation_logs['Validation/D_cost_real_valid'] = []
        validation_logs['Validation/G_cost_valid_adv'] = []
        validation_logs['Validation/G_cost_valid_l1'] = []
        validation_logs['Validation/G_cost_total_valid'] = []
    outputs_logs['G_cost_l1'] = []
    outputs_logs['G_cost_adv'] = []
    outputs_logs['G_cost_total'] = []

    # training
    for epoch in range(max(0, last_epoch), config.epochs):
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(train_loader):
            x, y, file = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

            for _ in range(config.rep_discriminator):
                optimizerD.zero_grad()
                # TRAIN D to recognize true RIRs
                Doutputs_real, _ = discriminator(torch.cat((y, x), dim=1))
                D_loss_real = torch.mean((Doutputs_real - 1.0) ** 2)  # L2 loss - we want them all to be 1
                outputs_logs['D_cost_real'].append(D_loss_real.cpu().detach().numpy())
                D_loss_real.backward()

                # TRAIN D to recognize aliased RIRs
                g_outputs = generator(x)
                Doutputs_fake, _ = discriminator(torch.cat((g_outputs, x), dim=1))
                D_loss_fake = torch.mean(Doutputs_fake ** 2)  # L2 loss - we want them all to be 0
                outputs_logs['D_cost_fake'].append(D_loss_fake.cpu().detach().numpy())
                D_loss_fake.backward()

                optimizerD.step()

            # Generator training
            optimizerG.zero_grad()
            Goutputs = generator(x)
            G_pair = torch.cat((Goutputs, x), dim=1)
            Doutputs_Gtrain, _ = discriminator(G_pair)

            G_adv_loss = config.G_adv_weight * torch.mean((Doutputs_Gtrain - 1.0) ** 2)
            outputs_logs['G_cost_adv'].append(G_adv_loss.cpu().detach().numpy())

            G_l1_loss = config.G_reg_weight * gen_l1_loss(Goutputs, y)
            outputs_logs['G_cost_l1'].append(G_l1_loss.cpu().detach().numpy())

            G_loss = G_adv_loss + G_l1_loss
            outputs_logs['G_cost_total'].append(G_loss.cpu().detach().numpy())

            G_loss.backward()
            optimizerG.step()

            # checkpointing
            if steps % save_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(chkpt_dir, steps)
                save_checkpoint(chkpt_dir,
                                checkpoint_path,
                                {'generator': generator.state_dict()},
                                remove_below_step=steps // 3)
                checkpoint_path = "{}/d_{:08d}".format(chkpt_dir, steps)
                save_checkpoint(chkpt_dir,
                                checkpoint_path,
                                {'discriminator': discriminator.state_dict(),
                                 'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict(),
                                 'steps': steps,
                                 'epoch': epoch},
                                remove_below_step=steps // 3)

            # Wandb summary logging
            if steps % summary_interval == 0:
                generator.eval()

                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                pearson = cos(y - y.mean(dim=-1, keepdim=True),
                              Goutputs - Goutputs.mean(dim=-1, keepdim=True))[0]

                log_dict = {key: outputs_logs[key][-1] for key in outputs_logs.keys()}
                log_dict.update({'steps': steps, 'pearson_corr_coeff': pearson})
                if use_wandb:
                    wandb.log(log_dict)

                # plot logs
                if steps % config.validation_interval == 0:  # and steps != 0:

                    ax1 = plot_frf(y_truth=y[0, 0, :].cpu().detach(),
                                   y_pred=x[0, 0, :].cpu().detach(),
                                   title='train input responses')
                    ax2 = plot_frf(y_truth=y[0, 0, :].cpu().detach(),
                                   y_pred=Goutputs[0, 0, :].cpu().detach(),
                                   title='train output responses')
                    if use_wandb:
                        wandb.log({"FRF_input_chart_train": wandb.Image(ax1)})
                        wandb.log({"FRF_output_chart_train": wandb.Image(ax2)})
                    plt.close('all')

                    ax3 = plot_rir(y_truth=y[0, 0, :].cpu().detach(),
                                   y_pred=x[0, 0, :].cpu().detach(),
                                   title='train input rir',
                                   y_pred_lab='Plane Wave Reconstruction RIR (Input)')
                    ax4 = plot_rir(y_truth=y[0, 0, :].cpu().detach(),
                                   y_pred=Goutputs[0, 0, :].detach().cpu(),
                                   title='train output rir',
                                   y_pred_lab='GAN Reconstruction RIR (Input)')
                    if use_wandb:
                        wandb.log({"RIR_eval_in_chart_train": wandb.Image(ax3)})
                        wandb.log({"RIR_eval_out_chart_train": wandb.Image(ax4)})
                    plt.close('all')

                    if validate:
                        cos_val = nn.CosineSimilarity(dim=-1, eps=1e-6)
                        with torch.no_grad():
                            try:
                                x_valid, y_valid = next(valid_iter)
                            except StopIteration:
                                valid_iter = iter(valid_loader)
                                x_valid, y_valid = next(valid_iter)

                            if cuda:
                                x_valid = x_valid.unsqueeze(1).cuda()
                                y_valid = y_valid.unsqueeze(1).cuda()

                            D_real_valid, _ = discriminator(torch.cat((x_valid, y_valid), dim=1) )
                            D_loss_real_valid = torch.mean((D_real_valid - 1.0) ** 2)  # L2 loss - we want them all to be 1
                            validation_logs['Validation/D_cost_real_valid'].append(D_loss_real_valid.cpu().numpy())


                            fake_valid = generator(x_valid)
                            Doutputs_fake_valid, _ = discriminator(torch.cat((fake_valid, y_valid), dim=1))
                            D_loss_fake_valid = torch.mean(Doutputs_fake_valid ** 2)  # L2 loss - we want them all to be 0
                            validation_logs['Validation/D_cost_fake_valid'].append(D_loss_fake_valid.cpu().numpy())

                            G_adv_loss_valid = config.G_adv_weight * torch.mean((Doutputs_fake_valid - 1.0) ** 2)
                            validation_logs['Validation/G_cost_valid_adv'].append(G_adv_loss_valid.cpu().numpy())

                            G_l1_loss_valid = config.G_reg_weight * gen_l1_loss(fake_valid, y_valid)
                            validation_logs['Validation/G_cost_valid_l1'].append(G_l1_loss_valid.cpu().numpy())

                            G_loss_valid = G_adv_loss_valid + G_l1_loss_valid
                            validation_logs['Validation/G_cost_total_valid'].append(G_loss_valid.cpu().numpy())

                            log_dict = {key: validation_logs[key][-1] for key in validation_logs.keys()}
                            cos_sim = cos_val(fake_valid, y_valid)
                            log_dict.update({'steps': steps, 'cos_sim': cos_sim})
                            if use_wandb:
                                wandb.log(log_dict)


                        ax1 = plot_frf(y_truth=y_valid[0, 0, :].cpu(),
                                       y_pred=x_valid[0, 0, :].cpu(),
                                       title='validation input responses')
                        ax2 = plot_frf(y_truth=y_valid[0, 0, :].cpu(),
                                       y_pred=fake_valid[0, 0, :].detach().cpu(),
                                       title='validation output responses')
                        if use_wandb:
                            wandb.log({"Validation/FRF_input_chart_validation": wandb.Image(ax1)})
                            wandb.log({"Validation/FRF_output_chart_validation": wandb.Image(ax2)})
                        plt.close('all')

                        ax3 = plot_rir(y_truth=y_valid[0, 0, :].cpu(),
                                       y_pred=x_valid[0, 0, :].cpu(),
                                       title='validation input rir',
                                       y_pred_lab='Plane Wave Reconstruction RIR (Input)')
                        ax4 = plot_rir(y_truth=y_valid[0, 0, :].cpu(),
                                       y_pred=fake_valid[0, 0, :].detach().cpu(),
                                       title='validation output rir',
                                       y_pred_lab='GAN Reconstruction RIR (Input)')
                        if use_wandb:
                            wandb.log({"Validation/RIR_eval_in_chart_valid": wandb.Image(ax3)})
                            wandb.log({"Validation/RIR_eval_out_chart_valid": wandb.Image(ax4)})
                        plt.close('all')

            generator.train()
            # print train info
            print_training_stats(epoch, epochs, total_batches, outputs_logs, i, start_time=start)
            steps += 1


if __name__ == '__main__':
    train()
