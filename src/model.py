"""
AdapterEM: Pre-trained Language Model Adaptation for Generalized Entity Matching using Adapter-tuning
Authors: John Bosco Mugeni, Steven Lynden, Toshiyuki Amagasa & Matono Akiyohi
Institute(s): University of Tsukuba (ibaraki, Japan) & National Institute Of Science & Technology (Tokyo Waterfront, Japan).

Accepted: 27th International Database Engineering And Applications Conference (IDEAS 2023) 
"""


import torch.nn as nn
from transformers.adapters import AdapterConfig
import numpy as np
from args import AdapterEMArgs
from transformers import AutoAdapterModel



plms = { 'bert':'bert-base-uncased' }


class AdapterEM(nn.Module):
    def __init__(self, args: AdapterEMArgs, lm_name='bert'):
        super().__init__()

        # load a pre-trained language model
        self.model = AutoAdapterModel.from_pretrained(plms[lm_name])
        
        # adapter configuration
        config = AdapterConfig.load(
            'houlsby',
            reduction_factor=args.adapter_size,
            non_linearity=args.non_linearity
        )

        if args.adapter_setup == "task_only":
            # if no pre-trained lanuage adapter specified, then add a new task adapter
            self.model.add_adapter("matching", config) # add a new task adapter
            self.model.train_adapter("matching") # freeze model theta's, turn on adapter
            self.model.set_active_adapters(['matching']) # activate for forward pass 
            self.model.add_classification_head(
                "matching", 
                num_labels=2,
                layers=1, 
                use_pooler=True
            ) 

        # stack a task adapter on top of snli language adapter from hub if specified                            
        elif args.adapter_setup == "snli_plus_task":
            snli_adapter = self.model.load_adapter(
                "adapters/snli_adapter",  
                with_head=False,
                config=config
            ) # language adapter name                                    
            self.model.add_adapter("matching", config) # add task adapter
            self.model.train_adapter("matching") # freeze model theta's, turn on adapter
            self.model.set_active_adapters([snli_adapter,'matching']) # stack adapters
            self.model.add_classification_head(
                "matching", 
                num_labels=2,
                layers=1, # linear layers before final classifier
                use_pooler=True
            ) # add a classification head

        # langauge modeling adapter @ 20% masking probability    
        elif args.adapter_setup == "tapt_20":
            tapt_adapter = self.model.load_adapter(
                f"adapters/20_masking/tapt_inv_{args.data_name}",
                 config=config, 
                 with_head=False
            ) # add tapt pre-trained adapter
            self.model.add_adapter("matching", config) # add  task adapter
            self.model.train_adapter(["matching"]) 
            self.model.set_active_adapters([tapt_adapter,'matching']) # activate for forward pass 
            self.model.add_classification_head(
                 "matching", 
                  num_labels=2,
                  layers=1, # linear layers before final classifier
                  use_pooler=True
            ) # add a classification head
        
        # langauge modeling adapter @ 40% masking probability
        elif args.adapter_setup == "tapt_40":
            # use MLM task adaptive pre-trained adapter i.e. 40% masking probability
            tapt_adapter = self.model.load_adapter(
                f"MLM_trained/tapt_inv_{args.data_name}", 
                config=config, 
                with_head=False
            ) # add tapt pre-trained adapter
            self.model.add_adapter("matching", config) 
            self.model.train_adapter(["matching"]) 
            self.model.set_active_adapters([tapt_adapter,'matching']) # activate for forward pass 
            self.model.add_classification_head(
                "matching", 
                num_labels=2,
                layers=1, # linear layers before final classifier
                use_pooler=True
            ) 

        if args.unfreeze_model:
            self.model.freeze_model(freeze=False)
        print(f"active_adapters: {self.model.active_adapters}")
        
        # count the number of trainable model parameters
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"trainable_params: {num_trainable_params}")

    def save_all_adapters(self, path, save_head=False):
        """Save entity matching adapters."""
        return self.model.save_all_adapters(save_directory=path, with_head=save_head)

    def forward(self, x):
        """Forward function of the model for classification."""
        logits = self.model(x).logits
        return logits



