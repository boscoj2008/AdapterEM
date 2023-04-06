"""
AdapterEM: Pre-trained Language Model Adaptation for Generalized Entity Matching using Adapter-tuning
Authors: John Bosco Mugeni, Steven Lynden, Toshiyuki Amagasa & Matono Akiyohi
Institute(s): University of Tsukuba (ibaraki, Japan) & National Institute Of Science & Technology (Tokyo Waterfront, Japan).
Accepted: 27th International Database Engineering And Applications Conference (IDEAS 2023) 


- Modifications based on PromptEM paper https://arxiv.org/pdf/2207.04802.pdf
"""

import argparse
import logging
 
 
class AdapterEMArgs:
    def __init__(self, args, data_name: str) -> None:
        self.seed = args.seed
        self.device = args.device
        self.model_name_or_path = args.model_name_or_path
        self.model_type = self.model_name_or_path.split("/")[-1].split("-")[0]
        self.batch_size = args.batch_size
        self.text_summarize = args.text_summarize
        self.learning_rate = args.lr
        self.max_length = args.max_length
        self.add_token = args.add_token
        self.data_name = data_name
        self.adapter_setup = args.adapter_setup
        self.non_linearity = args.non_linearity
        self.k = args.k
        self.num_iter = args.num_iter
        self.save_model = args.save_model
        self.unfreeze_model = args.unfreeze_model
        self.epochs = args.epochs
        self.adapter_size = args.adapter_size
  


    def __str__(self) -> str:
        return f"[{', '.join((f'{k}:{v}' for (k, v) in self.__dict__.items()))}]"

    def log(self):
        logging.info("====AdapterEM Args====")
        for (k, v) in self.__dict__.items():
            logging.info(f"{k}: {v}")


def int_or_float(value):
    try:
        value = int(value)
        return value
    except ValueError:
        try:
            value = float(value)
            return value
        except ValueError:
            return None


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path", "-model", type=str, default="bert")
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4, help="(learning rate) lr")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_token", default=True)
    parser.add_argument("--data_name", "-d", type=str, choices=["rel-heter", "rel-text", "semi-heter", "semi-homo", "semi-rel", "semi-text-c", "semi-text-w", "geo-heter" ,"all"], default="all")
    parser.add_argument("--adapter_setup", "-a_setup", type=str, choices=["snli_plus_task","task_only", "tapt_20", "tapt_40"], default='task_only')
    parser.add_argument("--non_linearity", type=str, choices=["swish","relu"], default="swish")
    parser.add_argument("--num_iter", "-ni", type=int, default=1)
    parser.add_argument("--k", "-k", type=int_or_float, default=0.10, help="(train rate) k")
    parser.add_argument("--text_summarize", "-ts", action="store_true")
    parser.add_argument("--save_model", "-save", action="store_true", default=False)
    parser.add_argument("--unfreeze_model", "-um", action="store_true", default=False)
    parser.add_argument("--epochs", "-n", type=int, default=20)
    parser.add_argument("--adapter_size", "-a_size", type=int, default=2)

    # Parse the arguments and return them
    args = parser.parse_args()
    return args


def parse_em_args(args, data_name) -> AdapterEMArgs:
    return AdapterEMArgs(args, data_name)
