import sys
sys.path.append('./jingwei_adversarial_ec2vae')
import json
import torch
torch.cuda.current_device()
import os
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
from collections import OrderedDict
import platform
from vae_triAd import ensembleModel
from jingwei_data_loader import MusicArrayLoader
from ruihan_data_loader import MusicArrayLoaderMusicArrayLoader_ruihan

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function_vae(recon,
                  recon_rhythm,
                  target_tensor,
                  rhythm_target,
                  distribution_1,
                  distribution_2,
                  beta=.1):
    CE1 = F.nll_loss(
        recon.view(-1, recon.size(-1)),
        target_tensor,
        reduction='mean')
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='mean')
    normal1 = std_normal(distribution_1.mean.size())
    normal2 = std_normal(distribution_2.mean.size())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    return CE1 + CE2 + beta * (KLD1 + KLD2), CE1 + CE2, KLD1 + KLD2

def loss_function_discr(chord_prediction, target_chord):
    chord_prediction.view(-1, chord_prediction.size(-1))
    target_chord.view(-1, target_chord.size(-1))
    criterion = torch.nn.BCELoss(weight=None, reduction='mean')
    CE_chord = criterion(chord_prediction, target_chord)
    return CE_chord


def train_vae(model, train_dataloader, epoch, loss_function_vae, loss_function_discr, optimizer, writer, args, beta=0.1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_kl = AverageMeter()
    losses_chord = AverageMeter()
    end = time.time()

    model.train()
    for step, (batch, c) in enumerate(train_dataloader):    #batch : batch * 32 *142; c: batch * 32 * 12 
        encode_tensor = batch.float()
        c = c.float()
        rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
        rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
        rhythm_target = torch.from_numpy(rhythm_target).float()
        rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
        target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
        if torch.cuda.is_available():
            encode_tensor = encode_tensor.cuda()
            target_tensor = target_tensor.cuda()
            rhythm_target = rhythm_target.cuda()
            c = c.cuda()

        optimizer.zero_grad()
        (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
        distribution_1 = Normal(dis1m, dis1s)
        distribution_2 = Normal(dis2m, dis2s)
        loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
        loss_chord = loss_function_discr(chord_prediction, c)   #theoretical optimal value is ln2
        #loss = loss + beta*loss_chord
        loss.backward()
        losses.update(loss.item())
        losses_recon.update(l_recon.item())
        losses_kl.update(l_kl.item())
        losses_chord.update(loss_chord.item())
        torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1)
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        #if (step + 1) % 200 == 0:
        #    if args['decay'] > 0:
        #        scheduler.step()

        if (step + 1) % args['display'] == 0:
            print('---------------------------Training VAE----------------------------')
            for param in optimizer.param_groups:
                print('lr1: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
        writer.add_scalar('train_vae/loss_total-epoch', losses.avg, epoch*len(train_dataloader)+step)
        writer.add_scalar('train_vae/loss_recon-epoch', losses_recon.avg, epoch*len(train_dataloader)+step)
        writer.add_scalar('train_vae/loss_KL-epoch', losses_kl.avg, epoch*len(train_dataloader)+step)
        writer.add_scalar('train_vae/loss_chord-epoch', losses_chord.avg, epoch*len(train_dataloader)+step)

def train_discr(model, train_dataloader, epoch, loss_function_vae, loss_function_discr, optimizer_discr, optimizer_enc, writer, args, beta=0.1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_kl = AverageMeter()
    losses_chord = AverageMeter()
    end = time.time()

    model.train()
    for step, (batch, c) in enumerate(train_dataloader):    #batch : batch * 32 *142; c: batch * 32 * 12 
        encode_tensor = batch.float()
        c = c.float()
        rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
        rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
        rhythm_target = torch.from_numpy(rhythm_target).float()
        rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
        target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
        if torch.cuda.is_available():
            encode_tensor = encode_tensor.cuda()
            target_tensor = target_tensor.cuda()
            rhythm_target = rhythm_target.cuda()
            c = c.cuda()
        if step % 3 == 0:
            target = 'Enc+Discr'
            optimizer = optimizer_discr
            optimizer.zero_grad()
            (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
            loss_chord = loss_function_discr(chord_prediction, c)
            with torch.no_grad():
                distribution_1 = Normal(dis1m, dis1s)
                distribution_2 = Normal(dis2m, dis2s)
                loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
            loss_chord.backward()
            losses.update(loss.item())
            losses_recon.update(l_recon.item())
            losses_kl.update(l_kl.item())
            losses_chord.update(loss_chord.item())
            torch.nn.utils.clip_grad_norm_(model.chd_vae.parameters(), 1)
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
        else:
            target = 'Enc'
            optimizer = optimizer_enc
            optimizer.zero_grad()
            (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
            loss_chord = loss_function_discr(chord_prediction, 1-c)
            with torch.no_grad():
                distribution_1 = Normal(dis1m, dis1s)
                distribution_2 = Normal(dis2m, dis2s)
                loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
            loss_chord.backward()
            losses.update(loss.item())
            losses_recon.update(l_recon.item())
            losses_kl.update(l_kl.item())
            losses_chord.update(loss_chord.item())
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1)
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

        #if (step + 1) % 200 == 0:
        #    if args['decay'] > 0:
        #        scheduler.step()

        if (step + 1) % args['display'] == 0:
            print('---------------------------Training ' + target + ' ----------------------------')
            for param in optimizer.param_groups:
                print('lr1: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
        writer.add_scalar('train_discr/loss_total-epoch', losses.avg, epoch*len(train_dataloader)+step)
        writer.add_scalar('train_discr/loss_recon-epoch', losses_recon.avg, epoch*len(train_dataloader)+step)
        writer.add_scalar('train_discr/loss_KL-epoch', losses_kl.avg, epoch*len(train_dataloader)+step)
        writer.add_scalar('train_discr/loss_chord-epoch', losses_chord.avg, epoch*len(train_dataloader)+step)


def validation(model, val_dataloader, epoch, loss_function_vae, loss_function_discr, writer, args, beta=0.1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_kl = AverageMeter()
    losses_chord = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for step, (batch, c) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            encode_tensor = batch.float()
            c = c.float()
            rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
            rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
            rhythm_target = torch.from_numpy(rhythm_target).float()
            rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
            target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
            if torch.cuda.is_available():
                encode_tensor = encode_tensor.cuda()
                target_tensor = target_tensor.cuda()
                rhythm_target = rhythm_target.cuda()
                c = c.cuda()

            (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
            distribution_1 = Normal(dis1m, dis1s)
            distribution_2 = Normal(dis2m, dis2s)
            loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
            loss_chord = loss_function_discr(chord_prediction, c)
            loss = loss + beta*loss_chord
            losses.update(loss.item())
            losses_recon.update(l_recon.item())
            losses_kl.update(l_kl.item())
            losses_chord.update(loss_chord.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (step + 1) % args['display'] == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)

    writer.add_scalar('val/loss_total-epoch', losses.avg, epoch)
    writer.add_scalar('val/loss_recon-epoch', losses_recon.avg, epoch)
    writer.add_scalar('val/loss_KL-epoch', losses_kl.avg, epoch)
    writer.add_scalar('val/loss_chord-epoch', losses_chord.avg, epoch)
    return losses_recon.avg


def main():
    # some initialization
    with open('./jingwei_adversarial_ec2vae/model_config.json') as f:
        args = json.load(f)
    
    model = ensembleModel(130, args['hidden_dim'], 3, 12, args['pitch_dim'],
                args['rhythm_dim'], args['time_step'])

    """print('let\'s fine tune!')
    if platform.system() == 'Linux':
        params = torch.load(args['Linux_name'])
    else:
        params = torch.load(args['name'])
    model.load_state_dict(params['model_state_dict'])"""
    if args['resume']:
        logdir = 'log/Mon Jul  6 22-51-32 2020'
        save_dir = 'params/Mon Jul  6 22-51-32 2020'
        #load pretrained file
        pretrained_file_path = 'params/{}.pt'.format(args['name'])
        params = torch.load(pretrained_file_path)
        model.load_state_dict(params)
        """
        params = torch.load('params/model_parameters.pt')
        new_params = OrderedDict()
        for k, v in params.items():
            name = k[7:]
            new_params[name] = v
        model.load_state_dict(new_params)
        """
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(r'log/Mon Jul  6 22-51-32 2020/events.out.tfevents.1594047092.Zhao-Jingwei')
        ea.Reload()
        val_acc=ea.scalars.Items('val/loss_total-epoch')
        init_epoch = len(val_acc)
        val_loss_record = val_acc[-1].value
    else:
        run_time = time.asctime(time.localtime(time.time())).replace(':', '-')
        logdir = 'log/' + run_time[4:]
        save_dir = 'params/' + run_time[4:]
        if platform.system == 'Linux':
            logdir = os.path.join(args["Linux_log_save"], logdir)
            save_dir = os.path.join(args["Linux_log_save"], save_dir)
        else:
            logdir = os.path.join(args["log_save"], logdir)
            save_dir = os.path.join(args["log_save"], save_dir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        init_epoch = 0
        val_loss_record = 100
    writer = SummaryWriter(logdir)
    
    if args['if_parallel']:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        gpu_num = 2
    else:
        gpu_num = 1
    optimizer_vae = optim.Adam(model.vae.parameters(), lr=args['lr'])
    optimizer_discr = optim.Adam(model.chd_vae.parameters(), lr=args['lr'])
    optimizer_enc = optim.Adam(model.encoder.parameters(), lr=args['lr'])
    optimizer_full = optim.Adam(model.parameters(), lr=args['lr'])
    if args['decay'] > 0:
        scheduler_vae = MinExponentialLR(optimizer_vae, gamma=args['decay'], minimum=1e-5,)
        scheduler_vae.last_epoch = init_epoch - 1
        scheduler_discr = MinExponentialLR(optimizer_discr, gamma=args['decay'], minimum=1e-5,)
        scheduler_discr.last_epoch = init_epoch - 1
        scheduler_enc = MinExponentialLR(optimizer_enc, gamma=args['decay'], minimum=1e-5,)
        scheduler_enc.last_epoch = init_epoch - 1
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')

    if platform.system() == 'Linux':
        train_datalist = args['Linux_train_path']
        val_datalist = args['Linux_val_path']
        batch_size = args['Linux_batch_size']
    else:
        train_datalist = args['train_path']
        val_datalist = args['val_path']
        batch_size = args['batch_size']
    
    # end of initialization
    train_dataset_1 = MusicArrayLoader(train_datalist, 0.005)
    train_dataset_2 = MusicArrayLoader(train_datalist, 0.001)
    val_dataset = MusicArrayLoader(val_datalist, 0.001)
    for epoch in range(init_epoch, args['n_epochs']):
        train_dataset_1.shuffle_data()
        train_dataset_2.shuffle_data()
        #val_dataset.shuffle_data()
        train_dataloader_1 = DataLoader(train_dataset_1, batch_size = batch_size*gpu_num,
                                    shuffle = False, drop_last = False)
        train_dataloader_2 = DataLoader(train_dataset_2, batch_size = batch_size*gpu_num//4,
                                    shuffle = False, drop_last = False)

        train_vae(model, train_dataloader_1, epoch, loss_function_vae, loss_function_discr, optimizer_vae, writer, args, beta=0.1)
        if args['decay'] > 0:
            scheduler_vae.step()
        train_discr(model, train_dataloader_2, epoch, loss_function_vae, loss_function_discr, optimizer_discr, optimizer_enc, writer, args, beta=0.1)
        if args['decay'] > 0:
            scheduler_discr.step()
            scheduler_enc.step()
        if epoch % 10 == 0:
            val_dataloader = DataLoader(val_dataset, batch_size = batch_size*gpu_num*2,
                                shuffle = False, drop_last = False)
            val_loss = validation(model, val_dataloader, epoch, loss_function_vae, loss_function_discr, writer, args, beta=0.1)
            if val_loss < val_loss_record:
                checkpoint = save_dir + '/best_fitted_params.pt'
                torch.save({'epoch': epoch, 'model_state_dict': model.vae.cpu().state_dict(), 'model_full_state_dict': model.cpu().state_dict(), 'optimizer_state_dict': optimizer_full.state_dict()}, checkpoint)
                model.cuda()
                val_loss_record = val_loss

def train_ruihan(dl, args, optimizer, loss_function, writer, scheduler, step):
    batch, c = dl.get_batch(args['batch_size'])
    print(batch.shape, c.shape)
    encode_tensor = torch.from_numpy(batch).float()
    c = torch.from_numpy(c).float()
    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
    target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
        rhythm_target = rhythm_target.cuda()
        c = c.cuda()
    optimizer.zero_grad()
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
    distribution_1 = Normal(dis1m, dis1s)
    distribution_2 = Normal(dis2m, dis2s)
    loss = loss_function(
        recon,
        recon_rhythm,
        target_tensor,
        rhythm_target,
        distribution_1,
        distribution_2,
        step,
        beta=args['beta'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1
    print('batch loss: {:.5f}'.format(loss.item()))
    writer.add_scalar('batch_loss', loss.item(), step)
    if args['decay'] > 0:
        scheduler.step()
    dl.shuffle_samples()
    return step


def main_ruihan():
    # some initialization
    with open('model_config.json') as f:
        args = json.load(f)
    if not os.path.isdir('log'):
        os.mkdir('log')
    if not os.path.isdir('params'):
        os.mkdir('params')
    save_path = 'params/{}.pt'.format(args['name'])
    writer = SummaryWriter('log/{}'.format(args['name']))
    model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'],
                args['rhythm_dim'], args['time_step'])
    if args['if_parallel']:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    if args['decay'] > 0:
        scheduler = MinExponentialLR(optimizer, gamma=args['decay'], minimum=1e-5)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')
    # end of initialization

    step, pre_epoch = 0, 0
    model.train()
    dl = MusicArrayLoader_ruihan(args['data_path'], args['time_step'], 16)
    dl.chunking()

    while dl.get_n_epoch() < args['n_epochs']:
        step = train_ruihan(dl, args, optimizer, loss_function_vae, writer, scheduler, step)
        if dl.get_n_epoch() != pre_epoch:
            pre_epoch = dl.get_n_epoch()
            torch.save(model.cpu().state_dict(), save_path)
            if torch.cuda.is_available():
                model.cuda()
            print('Model saved!')


if __name__ == '__main__':
    main()

