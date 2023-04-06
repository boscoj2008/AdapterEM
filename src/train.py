import logging
import random
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AdapterEM
import torch
import numpy as np
from args import AdapterEMArgs
from data import EMDataset, TypeDataset

from utils import evaluate, statistic_of_current_train_set
import time


def train_plm(args: AdapterEMArgs, model, labeled_train_dataloader, optimizer, scaler):
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_total = []
    for batch in tqdm(labeled_train_dataloader):
        x, labels = batch
        x = torch.tensor(x).to(args.device)
        labels = torch.tensor(labels).to(args.device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total.append(loss.item())
    return np.array(loss_total).mean()


def eval_plm(args: AdapterEMArgs, model, data_loader, return_acc=False):
    model.eval()
    y_truth = []
    y_pre = []
    for batch in tqdm(data_loader):
        x, labels = batch
        x = torch.tensor(x).to(args.device)
        y_truth.extend(labels)
        with torch.no_grad():
            logits = model(x)
            logits = torch.argmax(logits, dim=1)
            logits = logits.cpu().numpy().tolist()
            y_pre.extend(logits)
    return evaluate(np.array(y_truth), np.array(y_pre), return_acc=return_acc)


class BestMetric:
    def __init__(self):
        self.valid_f1 = -1
        self.test_metric = None
        self.state_dict = None


def inner_train(args: AdapterEMArgs, model, optimizer, scaler, train_dataloader, valid_dataloader, test_dataloader,
                prompt=True):
    if prompt:
        loss = train_prompt(args, model, train_dataloader, optimizer, scaler)
    else:
        loss = train_plm(args, model, train_dataloader, optimizer, scaler)
    logging.info(f"loss: {loss}")
    if prompt:
        valid_p, valid_r, valid_f1 = eval_prompt(args, model, valid_dataloader)
    else:
        valid_p, valid_r, valid_f1 = eval_plm(args, model, valid_dataloader)
    logging.info(f"[Valid] Precision: {valid_p:.4f}, Recall: {valid_r:.4f}, F1: {valid_f1:.4f}")
    if prompt:
        test_p, test_r, test_f1 = eval_prompt(args, model, test_dataloader)
    else:
        test_p, test_r, test_f1 = eval_plm(args, model, test_dataloader)
    logging.info(f"[Test] Precision: {test_p:.4f}, Recall: {test_r:.4f}, F1: {test_f1:.4f}")
    return (valid_p, valid_r, valid_f1, test_p, test_r, test_f1)


def update_best(model, metric, best: BestMetric):
    valid_p, valid_r, valid_f1, test_p, test_r, test_f1 = metric
    if valid_f1 > best.valid_f1 or valid_f1 == best.valid_f1 and test_f1 > best.test_metric[2]:
        best.valid_f1 = valid_f1
        best.test_metric = (test_p, test_r, test_f1)
        best.state_dict = model.state_dict()


def train_and_update_best(args: AdapterEMArgs, model, optimizer, scaler, train_dataloader, valid_dataloader,
                          test_dataloader, best: BestMetric, prompt=True):
    metric = inner_train(args, model, optimizer, scaler, train_dataloader, valid_dataloader, test_dataloader,
                             prompt)
    update_best(model, metric, best)




def adapter_train(args: AdapterEMArgs, data: EMDataset):
    train_set = TypeDataset(data, "train")
    valid_set = TypeDataset(data, "valid")
    test_set = TypeDataset(data, "test")
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad)
    best = BestMetric()
    start_time = time.time()
    for iter in range(1, args.num_iter + 1):
        # train model
        model = AdapterEM(lm_name=args.model_name_or_path, args=args)
        model.to(args.device)
        optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
        scaler = GradScaler()
        siz, pos, neg, per, acc = statistic_of_current_train_set(data)
        logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
        for epoch in range(1, args.epochs + 1):
            logging.info(f"epoch#{epoch}")
            train_and_update_best(args, model, optimizer, scaler, train_loader, valid_loader, test_loader, best,
                                  prompt=False)
                                  
        p, r, f1 = best.test_metric
        logging.info(f"[Best model in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        
        end_time = time.time()
        delta = end_time - start_time
        logging.info(f'training_time: {delta}')
        p, r, f1 = best.test_metric
        logging.info(f"[Best in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        logging.info(f"training time: {delta}")
