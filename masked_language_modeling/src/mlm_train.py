"""
AdapterEM: Pre-trained Language Model Adaptation for Generalized Entity Matching using Adapter-tuning
Authors: John Bosco Mugeni, Steven Lynden, Toshiyuki Amagasa & Matono Akiyohi
Institute(s): University of Tsukuba (ibaraki, Japan) & National Institute Of Science & Technology (Tokyo Waterfront, Japan).

Accepted: 27th International Database Engineering And Applications Conference (IDEAS 2023) 
"""



import os
import time
import logging
import json
import csv
import warnings 
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import random

from tqdm import tqdm
from transformers import (
BertForMaskedLM,
AutoTokenizer,
AdamW,
)
from transformers.adapters import (
HoulsbyConfig,
HoulsbyInvConfig,
AutoAdapterModel,
)

from mlm_data import GEMData, GEMDataset
from utils import (
set_seed,
set_logger,
read_entities,
read_ground_truth_few_shot,
)
from args import (
parse_args,
parse_em_args,
)

set_logger("MLM_training") 

plms = {
        'bert':'bert-base-uncased',
        'bert_cased':'bert-base-cased'
       }

# comment line 53 for debugging mode as needed
warnings.filterwarnings(action='ignore')

def main():
    
    common_args = parse_args()
    dataset_name = common_args.data_name
    data = GEMData(dataset_name)
    args = parse_em_args(common_args, dataset_name)
    args.log()

    # read entities
    data.left_entities, data.right_entities = read_entities(dataset_name, args)

    # read ground truth few-shot data
    data.train_pairs, data.train_y, data.train_un_pairs, data.train_un_y = read_ground_truth_few_shot(
                                         f"data/{dataset_name}", ["train"], k=args.k, return_un_y=True)

    # extract left and right entities from training pairs
    left_data = [data.left_entities[pair[0]] for pair in data.train_pairs]
    right_data = [data.right_entities[pair[1]] for pair in data.train_pairs]

    # merge left and right training texts without labels
    combined = right_data + left_data


    
    # Load pre-trained BERT model and tokenizer
    model_name_or_path = args.model_name_or_path
    bert = BertForMaskedLM.from_pretrained(plms[model_name_or_path])
    tokenizer = AutoTokenizer.from_pretrained(plms[model_name_or_path])


    # Add and train an adapter for inverse tokenization
    adapter_name = f"tapt_inv_{dataset_name}"
    config = HoulsbyInvConfig()
    bert.add_adapter(adapter_name, config=config)
    bert.train_adapter([adapter_name])
    bert.set_active_adapters(adapter_name)
   
    
    # Prepare dataset
    inputs = tokenizer(
    text=combined,
    return_tensors='pt',
    max_length=args.max_length,
    truncation=True,
    padding='max_length'
              )

    # Create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand_like(inputs.input_ids.float())

    # Create mask array
    mask_arr = (rand < args.mlm_prob) & \
         (inputs.input_ids != 101) & \
         (inputs.input_ids != 102) & \
         (inputs.input_ids != 0)

    inputs['labels'] = inputs.input_ids.detach().clone()

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
                         torch.flatten(mask_arr[i].nonzero()).tolist()
                         )
        
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103  
    
    # move model to GPU/ CPU
    bert.to(args.device)
    dataset = GEMDataset(inputs)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # init optimizer
    optimizer = torch.optim.AdamW(bert.parameters(), lr=args.learning_rate)

    ## eval block
    def evaluate(bert, valid_loader):
        bert.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, leave=True, desc="runing eval ..."):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                outputs = bert(input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss = outputs.loss.item()
                eval_losses.append(eval_loss)
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
        logging.info(f"mlm eval loss: {avg_eval_loss:.4f}")
    
    # train block
    def train(bert, train_loader, valid_loader, epochs, optimizer):
        for epoch in range(1, epochs + 1):
            bert.train()
            train_losses = []
            for batch in tqdm(train_loader, leave=True, desc=f"Epoch {epoch}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                outputs = bert(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_losses.append(train_loss)
                tqdm.write(f"Train loss: {train_loss:.4f}")
            avg_train_loss = sum(train_losses) / len(train_losses)
        evaluate(bert, valid_loader) 
    start = time.time()
    # train masked language model 
    train(bert, train_loader, valid_loader, args.epochs, optimizer) 
    
    finish = time.time()
    delta = finish - start
    print(f'MLM train time:{delta:.4f} seconds')
    bert.save_all_adapters(save_directory="../adapters/{}_masking/{}".format(int(args.mlm_prob*100) ,dataset_name), with_head=False) # save adapter



if __name__ == "__main__":
    main()
      
