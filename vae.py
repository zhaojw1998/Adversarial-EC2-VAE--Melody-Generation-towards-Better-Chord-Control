import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np


class VAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 rhythm_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims,
                 n_step,
                 k=1000):
        super(VAE, self).__init__()
        self.gru_0 = nn.GRU(
            roll_dims + condition_dims,
            hidden_dims,
            batch_first=True,
            bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.grucell_0 = nn.GRUCell(z2_dims + rhythm_dims,
                                    hidden_dims)
        self.grucell_1 = nn.GRUCell(
            z1_dims + roll_dims + rhythm_dims + condition_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_0 = nn.Linear(z2_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_init_1 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, roll_dims)
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.rhythm_sample = None
        self.iteration = 0
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

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

    def rhythm_decoder(self, z):
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
                    out = self.rhythm_sample[:, i, :]
                else:
                    out = self._sampling(out)
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def final_decoder(self, z, rhythm, condition):
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
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def decoder(self, z1, z2, condition=None):
        rhythm = self.rhythm_decoder(z2)
        return self.final_decoder(z1, rhythm, condition)

    def forward(self, x, condition):
        if self.training:
            self.sample = x
            self.rhythm_sample = x[:, :, :-2].sum(-1).unsqueeze(-1)
            self.rhythm_sample = torch.cat((self.rhythm_sample, x[:, :, -2:]),
                                           -1)
            self.iteration += 1
        dis1, dis2 = self.encoder(x, condition)
        z1 = dis1.rsample()
        z2 = dis2.rsample()
        recon_rhythm = self.rhythm_decoder(z2)
        recon = self.final_decoder(z1, recon_rhythm, condition)
        output = (recon, recon_rhythm, dis1.mean, dis1.stddev, dis2.mean,
                  dis2.stddev)
        return output, z1

class discriminator(nn.Module):
    def __init__(self, z1_dims, condition_dims, hidden_dims, n_step):
        super(discriminator, self).__init__()
        self.z1_dims = z1_dims
        self.condition_dims = condition_dims
        self.hidden_dims = hidden_dims
        self.n_step = n_step
        self.eps = 1
        self.grucell_0 = nn.GRUCell(self.z1_dims + self.condition_dims, self.hidden_dims)
        self.linear_init_0 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(self.hidden_dims, self.condition_dims)
    
    def chord_classifier(self, z, condition_gt=None):
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
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = condition_gt[:, i, :]
                else:
                    out = (out >= 0.5).float()
            else:
                out = (out >= 0.5).float()
        return torch.stack(x, 1)

    def forward(self, z, condition):
        output = self.chord_classifier(z, condition)
        return output

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
        self.vae = VAE(roll_dims, hidden_dims, rhythm_dims, condition_dims, z1_dims, z2_dims, n_step, k)
        self.discr = discriminator(z1_dims, condition_dims, hidden_dims, n_step)

    def vae_model(self):
        return self.vae
    def discr_model(self):
        return self.discr

    def forward(self, x, condition):
        output, z1 = self.vae(x, condition)
        chord_prediction = self.discr(z1, condition)
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
    """
    model = ensembleModel(130, 1024, 3, 12, 128, 128, 32).cuda()
    x = torch.from_numpy(np.random.rand(8, 32, 130)).float().cuda()
    condition = torch.from_numpy(np.random.rand(8, 32, 12)).float().cuda()
    o, c = model(x, condition)
    print(o[0].shape, c.shape)
    """
    model = VAE(130, 1024, 3, 12, 128, 128, 32)
    params = torch.load('D:/Computer Music Research/Derek/params/Nov 30 01-37-16 2020/best_fitted_params.pt')['model_state_dict']
    from collections import OrderedDict
    renamed_params = OrderedDict()
    for k, v in params.items():
        name = '.'.join(k.split('.')[1:])
        renamed_params[name] = v
    model.load_state_dict(renamed_params)
    print('YES')
    model.cuda()
