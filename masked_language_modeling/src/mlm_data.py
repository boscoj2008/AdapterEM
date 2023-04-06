"""
 @Paper:- AdapterEM:Pre-trained Language Model Adaptation fo Generalized Entity Matching using Adapter-tuning
 @Authors:- John Bosco Mugeni, Steven Lynden, Toshiyuki Amagasa, Matono Akiyohi
 @Institute:- National Institute Of Science & Technology, Tokyo Waterfront, Japan.
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer


class GEMData:
    def __init__(self, data_type):
        self.data_type = data_type
        self.left_entities = []
        self.right_entities = []
        self.train_pairs = []
        self.train_y = []
        self.train_un_pairs = []
        # only used in test_pseudo_labels, will not be updated
        self.train_un_y = []
        self.valid_pairs = []
        self.valid_y = []
        self.test_pairs = []
        self.test_y = []
        self.ground_truth = set()

    def read_all_ground_truth(self, file_path):
        self.ground_truth = []
        for file in ["train", "valid", "test"]:
            with open(os.path.join(file_path, f"{file}.csv"), "r") as rd:
                for i, line in enumerate(rd.readlines()):
                    values = line.strip().split(',')
                    if int(values[2]) == 1:
                        self.ground_truth.append((int(values[0]), int(values[1])))
        self.ground_truth = set(self.ground_truth)



class GEMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
        
        
