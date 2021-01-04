import muspy
import numpy as np
import os
import torch

if not os.path.exists('data/Nottingham/'):
    os.makedirs('data/Nottingham/')
mnd = muspy.NottinghamDatabase("data/Nottingham/", download_and_extract=True)
mnd.convert()
#for idx, music in enumerate(mnd):   #220 4 tracks; 451 3 tempos
#    if not len(music.tempos) == 1:
#        print(idx, music.tempos)
dataset = mnd.to_pytorch_dataset(representation="pitch")
print(dataset[0].shape)