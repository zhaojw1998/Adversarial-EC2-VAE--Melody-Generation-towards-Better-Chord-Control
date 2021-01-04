import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np


class VAE_Encoder(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims):
        super(VAE_Encoder, self).__init__()
        self.gru_0 = nn.GRU(
            roll_dims + condition_dims,
            hidden_dims,
            batch_first=True,
            bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims

    def encoder(self, x, condition):
        # self.gru_0.flatten_parameters()
        x = torch.cat((x, condition), -1)
        x = self.gru_0(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        distribution_1 = Normal(mu[:, :self.z1_dims], var[:, :self.z1_dims])
        distribution_2 = Normal(mu[:, self.z1_dims:], var[:, self.z1_dims:])
        return distribution_1, distribution_2

    def forward(self, x, condition):
        dis1, dis2 = self.encoder(x, condition)
        z1 = dis1.rsample()
        z2 = dis2.rsample()
        return z1, z2, dis1, dis2

class VAE_Decoder_Rhythm(nn.Module):
    def __init__(self,
                 hidden_dims,
                 rhythm_dims,
                 z2_dims,
                 n_step,
                 k=1000):
        super(VAE_Decoder_Rhythm, self).__init__()
        self.grucell_0 = nn.GRUCell(z2_dims + rhythm_dims,
                                    hidden_dims)
        self.linear_init_0 = nn.Linear(z2_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, rhythm_dims)
        self.n_step = n_step
        self.eps = 1
        self.rhythm_dims = rhythm_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def rhythm_decoder(self, z, rhythm_sample, iteration):
        out = torch.zeros((z.size(0), self.rhythm_dims))
        out[:, -1] = 1.
        x = []
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, z], 1)
            hx = self.grucell_0(out, hx)
            out = F.log_softmax(self.linear_out_0(hx), 1)
            x.append(out)
            if self.training:  
                p = torch.rand(1).item()
                if p < self.eps:
                    out = rhythm_sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / (self.k + torch.exp(iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, z2, rhythm_sample, iteration):
        recon_rhythm = self.rhythm_decoder(z2, rhythm_sample, iteration)
        return recon_rhythm

class VAE_Decoder_Final(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 rhythm_dims,
                 condition_dims,
                 z1_dims,
                 n_step,
                 k=1000):
        super(VAE_Decoder_Final, self).__init__()
        self.grucell_1 = nn.GRUCell(
            z1_dims + roll_dims + rhythm_dims + condition_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_1 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, roll_dims)
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.eps = 1
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def final_decoder(self, z, rhythm, sample, iteration, condition):
        out = torch.zeros((z.size(0), self.roll_dims))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, rhythm[:, i, :], z, condition[:, i, :]], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_1(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                    (self.k + torch.exp(iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, z1, dis1, dis2, recon_rhythm, sample, iteration, condition):
        recon = self.final_decoder(z1, recon_rhythm, sample, iteration, condition)
        output = (recon, recon_rhythm, dis1.mean, dis1.stddev, dis2.mean,
                  dis2.stddev)
        return output

class VAE(nn.Module):
    def __init__(self, encoder, rhy_decoder, final_decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.rhy_decoder = rhy_decoder
        self.final_decoder = final_decoder
        self.sample = None
        self.rhythm_sample = None
        self.iteration = 0

    def forward(self, x, condition):
        if self.training:
            self.sample = x
            self.rhythm_sample = x[:, :, :-2].sum(-1).unsqueeze(-1)
            self.rhythm_sample = torch.cat((self.rhythm_sample, x[:, :, -2:]), -1)
            self.iteration += 1
        z1, z2, dis1, dis2 = self.encoder(x, condition)
        recon_rhythm = self.rhy_decoder(z2, self.rhythm_sample, self.iteration)
        output = self.final_decoder(z1, dis1, dis2, recon_rhythm, self.sample, self.iteration, condition)
        #output = (recon, recon_rhythm, dis1.mean, dis1.stddev, dis2.mean, dis2.stddev)
        return output

class discriminator(nn.Module):
    def __init__(self, z1_dims, condition_dims, hidden_dims, n_step, k=1000):
        super(discriminator, self).__init__()
        self.z1_dims = z1_dims
        self.condition_dims = condition_dims
        self.hidden_dims = hidden_dims
        self.n_step = n_step
        self.eps = 1
        self.grucell_0 = nn.GRUCell(self.z1_dims + self.condition_dims, self.hidden_dims)
        self.linear_init_0 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(self.hidden_dims, self.condition_dims)
        self.k = torch.FloatTensor([k])
        self.teacher_forcing = True
    
    def chord_classifier(self, z, iteration, condition_gt=None):
        out = torch.zeros((z.size(0), self.condition_dims))
        x = []
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, z], 1)
            hx = self.grucell_0(out, hx)
            out = torch.sigmoid(self.linear_out_0(hx))
            x.append(out)
            if self.teacher_forcing and self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = condition_gt[:, i, :]
                else:
                    out = (out >= 0.5).float()
                self.eps = self.k / (self.k + torch.exp(iteration / self.k))
            else:
                out = (out >= 0.5).float()
        return torch.stack(x, 1)

    def forward(self, z, condition, iteration):
        output = self.chord_classifier(z, iteration, condition)
        return output

class VAE_Chord(nn.Module):
    def __init__(self, encoder, discr):
        super(VAE_Chord, self).__init__()
        self.encoder = encoder
        self.discr = discr
    def forward(self, x, condition):
        z1, z2, dis1, dis2 = self.encoder(x, condition)
        chord_prediction = self.discr(z1, condition)
        return chord_prediction


class ensembleModel(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 rhythm_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims,
                 n_step, 
                 k=1000):
        super(ensembleModel, self).__init__()
        self.encoder = VAE_Encoder(roll_dims, hidden_dims, condition_dims, z1_dims, z2_dims)
        self.rhy_decoder = VAE_Decoder_Rhythm(hidden_dims, rhythm_dims, z2_dims, n_step, k=1000)
        self.final_decoder = VAE_Decoder_Final(roll_dims, hidden_dims, rhythm_dims, condition_dims, z1_dims, n_step, k=1000)
        
        self.discr = discriminator(z1_dims, condition_dims, hidden_dims, n_step)
        self.vae = VAE(self.encoder, self.rhy_decoder, self.final_decoder)
        self.chd_vae = VAE_Chord(self.encoder, self.discr)

        self.sample = None
        self.rhythm_sample = None
        self.iteration = 0

    def vae_model(self):
        return self.vae

    def discr_model(self):
        return self.chd_vae

    def vae_encoder(self):
        return self.encoder

    def vae_rhy_decoder(self):
        return self.rhy_decoder

    def vae_final_decoder(self):
        return self.final_decoder

    def chord_discriminator(self):
        return self.discr    

    def forward(self, x, condition):
        if self.training:
            self.sample = x
            self.rhythm_sample = x[:, :, :-2].sum(-1).unsqueeze(-1)
            self.rhythm_sample = torch.cat((self.rhythm_sample, x[:, :, -2:]), -1)
            self.iteration += 1
        z1, z2, dis1, dis2 = self.encoder(x, condition)
        recon_rhythm = self.rhy_decoder(z2, self.rhythm_sample, self.iteration)
        output = self.final_decoder(z1, dis1, dis2, recon_rhythm, self.sample, self.iteration, condition)
        chord_prediction = self.discr(dis1.mean, condition, self.iteration)
        return output, chord_prediction


if __name__ == '__main__':
    """
    #test scripts for discriminator
    dis = discriminator(128, 12, 1024, 32).cuda()
    z = torch.from_numpy(np.random.rand(8, 128)).float().cuda()
    condition_gt = torch.from_numpy(np.zeros((8, 32, 12))).float().cuda()
    condition_gt[:, 1] = 1
    condition_gt[:, 4] = 1
    condition_gt[:, 7] = 1
    dis.train()
    out = dis.rhythm_decoder(z, condition_gt)
    print(out.shape)
    """
    #test scripts for ensemble model
    model = ensembleModel(130, 1024, 3, 12, 128, 128, 32).cuda()
    x = torch.from_numpy(np.random.rand(8, 32, 130)).float().cuda()
    condition = torch.from_numpy(np.random.rand(8, 32, 12)).float().cuda()
    o, c, = model(x, condition)
    print(o[0].shape, c.shape)

    checkpoint = 'test.pt'
    torch.save({'model_state_dict': model.vae.cpu().state_dict()}, checkpoint)
    #print(model)

  