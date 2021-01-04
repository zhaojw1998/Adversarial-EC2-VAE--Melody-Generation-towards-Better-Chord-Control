import os
import numpy as np
import torch.utils.data as data

class MusicArrayLoader(data.Dataset):
    def __init__(self, data_path, ratio):
        self.ratio = ratio
        self.data_full = np.load(data_path, allow_pickle=False)
        #print(self.data_full.shape)
        np.random.shuffle(self.data_full)
        self.data = self.data_full[:int(self.ratio*len(self.data_full))]
        #print(self.datalist[:10])

    def __getitem__(self, index):
        item = self.data[index]
        melody = item[:, :130]
        chord = item[:, 130:]
        return melody, chord

    def __len__(self):
        return self.data.shape[0]
    
    def shuffle_data(self):
        np.random.shuffle(self.data_full)
        self.data = self.data_full[:int(self.ratio*len(self.data_full))]

if __name__ == '__main__':
    loader = MusicArrayLoader('D:/nottingham_32beat-split_npy_train-val-split/nottingham_32beat-split_val.npy', 0.2)