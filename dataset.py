import torch
from torch.utils.data import Dataset
from utils import *
from encoder import EncoderandDecoder

class LanguageModelDataset(Dataset):
    def __init__(self, corpora_path: str, block_size: int, dataset_type='train', transform=None):
        
        self.unique_characters = generate_character_lexicon(corpora_path=corpora_path)
        self.encoder_decoder = EncoderandDecoder(list_of_tokens=self.unique_characters)
        self.block_size = block_size

        # Memory consuming. Okay for small corpora but problematic when it is big.
        with open(corpora_path) as f:
            self.data = f.read()

        self.data = self.data[:int(len(self.data)*0.8)] if dataset_type == 'train' else self.data[int(len(self.data)*0.8):]
        self.data_to_sequential_block_sizes(block_size=block_size)
        self.transform = transform

        del self.data

    def __len__(self):
        return len(self.xs)
    
    def data_to_sequential_block_sizes(self, block_size: int):
        
        self.xs, self.ys= [], []    
        assert len(self.data) >= block_size 

        for i in range(len(self.data)-block_size):
            x_chunk = self.data[i:i+block_size]
            y_chunk = self.data[i+1:i+block_size+1]
            self.xs.append(self.encoder_decoder.rudimentary_encoder(x_chunk))
            self.ys.append(self.encoder_decoder.rudimentary_encoder(y_chunk))

    def __getitem__(self, idx):
        x_i, y_i = torch.tensor(self.xs[idx], dtype=torch.long), torch.tensor(self.ys[idx], dtype=torch.long)
        if self.transform:
            x_i, y_i = self.transform(x_i, y_i)
        return {'input': x_i, 'target': y_i}