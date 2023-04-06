"""
 @Paper:- AdapterEM:Pre-trained Language Model Adaptation fo Generalized Entity Matching using Adapter-tuning
 @Authors:- John Bosco Mugeni, Steven Lynden, Toshiyuki Amagasa, Matono Akiyohi
 @Institute:- National Institute Of Science & Technology, Tokyo Waterfront, Japan.
 
 - Modifications have been made in part to the training code of PromptEM paper https://arxiv.org/pdf/2207.04802.pdf to devise AdapterEM.
"""

import argparse
import logging


class MLMArgs:
    def __init__(self, args, data_name: str) -> None:
        self.seed = args.seed
        self.device = args.device
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.max_length = args.max_length
        self.add_token = args.add_token
        self.data_name = data_name
        self.text_summarize = args.text_summarize
        self.model_name_or_path = args.model_name_or_path
        self.k = args.k
        self.epochs = args.epochs
        self.mlm_prob = args.mlm_prob

        
  


    def __str__(self) -> str:
        return f"[{', '.join((f'{k}:{v}' for (k, v) in self.__dict__.items()))}]"

    def log(self):
        logging.info("====MLM-BERT====")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path","-model", type=str, default="bert")
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="model learning rate")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_token", default=True)
    parser.add_argument("--data_name", "-d", type=str,
                        choices=["rel-heter", "rel-text", "semi-heter", "semi-homo", "semi-rel", "semi-text-c",
                                 "semi-text-w", "geo-heter" ,"all"], default="all")

    parser.add_argument("--k", "-k", type=int_or_float, default=1.0)
    parser.add_argument("--epochs", "-n", type=int, default=3)
    parser.add_argument("--mlm_prob", type=float, default=0.20)
    parser.add_argument("--text_summarize", '-t', default=False)

  

    args = parser.parse_args()
    return args


def parse_em_args(args, data_name) -> MLMArgs:
    return MLMArgs(args, data_name)
