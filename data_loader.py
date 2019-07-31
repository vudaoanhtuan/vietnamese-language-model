import os
import pickle

from tokenizer import VNTokenizer

import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

def build_tokenizer_from_word_list(word_list, save_file=None):
    tokenizer = VNTokenizer(word_list)
    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump(tokenizer, f)
    return tokenizer

def build_tokenizer_from_file(word_list_file_path, save_file=None):
    if not os.path.isfile(word_list_file_path):
        raise Exception("Invalid path")
    with open(word_list_file_path) as f:
        word_list = f.read().split('\n')[:-1]
    tokenizer = build_tokenizer_from_word_list(word_list, save_file)
    return tokenizer

def load_tokenizer(file_path):
    if not os.path.isfile(file_path):
        raise Exception("Invalid path")
    with open(file_path, 'rb') as f:
        tokenizer = pickle.load(f)
        return tokenizer

def build_data_iter(file_path, tokenizer, batch_size=32, shuffle=True):
    if not os.path.isfile(file_path):
        raise Exception("Invalid path")
    with open(file_path) as f:
        data = f.read().split('\n')[:-1]
    X_all = tokenizer.texts_to_sequences(data, add_bos=True, add_eos=True)
    X_all = tokenizer.pad_sequences(X_all)
    X = X_all[:-1]
    y = X_all[1:]

    X = torch.tensor(X).long()
    y = torch.tensor(y).long()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
