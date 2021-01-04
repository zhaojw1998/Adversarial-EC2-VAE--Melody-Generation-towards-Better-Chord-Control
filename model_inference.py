import pretty_midi as pyd
import torch
import os
import json
import numpy as np
from vae import ensembleModel, VAE
from midi_interface_mono_and_chord import midi_interface_mono_and_chord
            
with open('jingwei_adversarial_ec2vae/model_config.json') as f:
    args = json.load(f)
weight_path = args['name']
processor = midi_interface_mono_and_chord()
model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step'])
#model = ensembleModel(130, args['hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step']).cuda()
params = torch.load(weight_path)['model_state_dict']
from collections import OrderedDict
renamed_params = OrderedDict()
for k, v in params.items():
    name = '.'.join(k.split('.')[1:])
    renamed_params[name] = v
model.load_state_dict(renamed_params)
model.cuda()
#model.load_state_dict(torch.load(weight_path)['model_state_dict'])
model.eval()
data_root = 'D:/Download/Program/xml/mid'
save_root = './jingwei_adversarial_ec2vae/qinying_test'
for item in os.listdir(data_root):
    batch, batch_with_NewChord, tempo = processor.load_single(os.path.join(data_root, item))
    print(batch.shape, tempo)
    melody = torch.from_numpy(batch[:, :, :130]).cuda().float()
    chord = torch.from_numpy(batch[:, :, 130:]).cuda().float()
    changed_chord = torch.from_numpy(batch_with_NewChord[:, :, 130:]).cuda().float()
    distr_pitch, distr_rhythm = model.encoder(melody, chord)
    z_pitch = distr_pitch.mean
    z_rhythm = distr_rhythm.mean
    recon_rhythm = model.rhythm_decoder(z_rhythm)
    recon = model.final_decoder(z_pitch, recon_rhythm, changed_chord)
    recon = recon.detach().cpu()[0]
    idx = recon.max(1)[1]
    out = torch.zeros_like(recon)
    arange = torch.arange(out.size(0)).long()
    out[arange, idx] = 1
    print(out.shape)
    midi_ReGen = processor.midiReconFromNumpy(np.concatenate((out, batch_with_NewChord[0, :, 130:]), axis=-1), tempo)
    midi_ReGen.write(os.path.join(save_root, item+'_+4.mid'))

    """Melody_A = np.load(np.random.choice(data_list))
    Melody_B = np.load(np.random.choice(data_list))
    numpy_to_midi_with_condition(Melody_A[:, :130], Melody_A[:, 130:], time_step = 0.125, output='test_A.mid')
    numpy_to_midi_with_condition(Melody_B[:, :130], Melody_B[:, 130:], time_step = 0.125, output='test_B.mid')

    Melody_A = torch.from_numpy(Melody_A).cuda().float()
    Melody_B = torch.from_numpy(Melody_B).cuda().float()
    melody_A = Melody_A[:, :130][np.newaxis, :, :]
    chord_A = Melody_A[:, 130:][np.newaxis, :, :]
    melody_B = Melody_B[:, :130][np.newaxis, :, :]
    chord_B = Melody_B[:, 130:][np.newaxis, :, :]

    distr_A_pitch, distr_A_rhythm = model.encoder(melody_A, chord_A)
    distr_B_pitch, distr_B_rhythm = model.encoder(melody_B, chord_B)
    z_A_pitch = distr_A_pitch.rsample()
    z_B_rhythm = distr_B_rhythm.rsample()
    recon_rhythm = model.rhythm_decoder(z_B_rhythm)
    #Reconstruct with chord of A
    recon = model.final_decoder(z_A_pitch, recon_rhythm, chord_A)
    recon = recon.detach().cpu()[0]
    idx = recon.max(1)[1]
    out = torch.zeros_like(recon)
    arange = torch.arange(out.size(0)).long()
    out[arange, idx] = 1
    numpy_to_midi_with_condition(out.numpy(), chord_A.detach().cpu().numpy()[0], time_step = 0.125, output='test_analogy_with_chord_A.mid')
    #Reconstruct with chord of B
    recon = model.final_decoder(z_A_pitch, recon_rhythm, chord_B)
    recon = recon.detach().cpu()[0]
    idx = recon.max(1)[1]
    out = torch.zeros_like(recon)
    arange = torch.arange(out.size(0)).long()
    out[arange, idx] = 1
    numpy_to_midi_with_condition(out.numpy(), chord_B.detach().cpu().numpy()[0], time_step = 0.125, output='test_analogyy_with_chord_B.mid')"""