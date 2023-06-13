# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:58:18 2023

@author: Shahir
"""


import os
import time
import pickle
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import copy
from tqdm import tqdm

from cs168_project.utils import get_timestamp_str


class ModelTracker:
    def __init__(
            self,
            root_dir: str) -> None:

        experiment_dir = "Experiment {}".format(get_timestamp_str())
        self.save_dir = os.path.join(root_dir, experiment_dir)
        self.best_model_metric = float('-inf')
        self.record_per_epoch = {}

    def update_info_history(
            self,
            epoch: int,
            info: Any) -> None:

        os.makedirs(self.save_dir, exist_ok=True)
        self.record_per_epoch[epoch] = info
        fname = "Experiment Epoch Info History.pckl"
        with open(os.path.join(self.save_dir, fname), 'wb') as f:
            pickle.dump(self.record_per_epoch, f)

    def update_model_weights(
            self,
            epoch: int,
            model_state_dict: dict,
            metric: Optional[float] = None,
            save_best: bool = True,
            save_latest: bool = True,
            save_current: bool = False) -> None:

        os.makedirs(self.save_dir, exist_ok=True)
        update_best = metric is None or metric > self.best_model_metric
        if update_best and metric is not None:
            self.best_model_metric = metric

        if save_best and update_best:
            torch.save(
                model_state_dict, os.path.join(
                    self.save_dir,
                    "Weights Best.pckl"))
        if save_latest:
            torch.save(
                model_state_dict, os.path.join(
                    self.save_dir,
                    "Weights Latest.pckl"))
        if save_current:
            torch.save(
                model_state_dict, os.path.join(
                    self.save_dir,
                    "Weights Epoch {} {}.pckl".format(epoch, get_timestamp_str())))


def make_optimizer(
        params_to_update: list[torch.Tensor],
        lr: float = 0.001,
        weight_decay: float = 1e-9,
        clip_grad_norm: bool = False) -> optim.Optimizer:

    optimizer = optim.AdamW(
        params_to_update,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=True)

    if clip_grad_norm:
        nn.utils.clip_grad_norm_(params_to_update, 3.0)

    return optimizer


def get_lr(
        optimizer: optim.Optimizer) -> None:

    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_optimizer_lr(
        optimizer: optim.Optimizer,
        lr: float) -> None:

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_scheduler(
        optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, threshold=0.025, patience=10, cooldown=5, min_lr=1e-6, verbose=True)

    return scheduler


def train_ae_model(
    device,
    model,
    dataloaders,
    optimizer,
    criterion,
    save_dir,
    labeled=False,
    lr_scheduler=None,
    save_model=False,
    save_model_all_epochs=False,
    save_log=False,
    num_epochs=1,
    train_batch_multiplier=1):
    
    start_time = time.time()
    
    tracker = ModelTracker(save_dir)
    
    train_batch_multiplier = int(train_batch_multiplier)

    epoch_phases = ['train', 'test']
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        train_loss_info = {}
        
        # Each epoch has a training and validation phase
        for epoch_phase in epoch_phases:
            if epoch_phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            if epoch_phase == 'train':
                running_loss = 0.0   
                running_count = 0
                batch_count = 0
                
                train_loss_record = []
                # Iterate over data.
                # TQDM has nice progress bars
                pbar = tqdm(dataloaders['train'])
                for inputs in pbar:
                    inputs = inputs.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    with torch.set_grad_enabled(True):
                        # Get model outputs and calculate loss 
                        loss = criterion.main_loss(model, inputs, device)
                        
                        # loss parts are for debugging purposes
                        loss_parts = loss
                        try:
                            iter(loss_parts)
                        except TypeError:
                            loss_parts = [loss_parts]
                        
                        loss = sum(loss_parts)
                        train_loss_record.append([n.detach().item() for n in loss_parts])
                        
                        loss.backward()
                        if batch_count == 0:
                            optimizer.step()
                            batch_count = train_batch_multiplier
                    
                        batch_count -= 1
                        
                    running_loss += loss.detach().item() * inputs.size(0)
                    running_count += inputs.size(0)
                    epoch_loss = running_loss / running_count
                    
                    loss_fmt = "{:.4f}"
                    desc = "Avg. Loss: {}, Total Loss: {}, Loss Parts: [{}]".format(loss_fmt.format(epoch_loss),
                                                                                    loss_fmt.format(sum(loss_parts)),
                                                                                    ", ".join(loss_fmt.format(n.item()) for n in loss_parts))
                    pbar.set_description(desc)
                    
                    del loss, loss_parts
                    
                print("Training Loss: {:.4f}".format(epoch_loss))
                train_loss_info['loss'] = train_loss_record
            
            elif epoch_phase == 'test':
                if labeled:
                    pass
            
            torch.cuda.empty_cache()
        
        if save_model:
            model_weights = copy.deepcopy(model.state_dict())
            tracker.update_model_weights(epoch, model_weights,
                                         save_current=save_model_all_epochs)
            info = {'train_loss_history': train_loss_info}
        
        if save_log:
            tracker.update_info_history(epoch, info)
        
        print()
        
        if lr_scheduler:
            lr_scheduler.step()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, tracker


def load_weights(
        model: nn.Module,
        weights_fname,
        map_location=None) -> nn.Module:

    model.load_state_dict(torch.load(weights_fname, map_location=map_location))

    return model


def save_training_session(
        model: nn.Module,
        optimizer: optim.Optimizer,
        sessions_save_dir: str) -> str:

    sub_dir = "Session {}".format(get_timestamp_str())
    sessions_save_dir = os.path.join(sessions_save_dir, sub_dir)
    os.makedirs(sessions_save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(
        sessions_save_dir, "Model State.pckl"))
    torch.save(optimizer.state_dict(), os.path.join(
        sessions_save_dir, "Optimizer State.pckl"))
    print("Saved session to", sessions_save_dir)

    return sessions_save_dir


def load_training_session(
        model: nn.Module,
        optimizer: optim.Optimizer,
        session_dir: str,
        update_models: bool = True,
        map_location: Optional[torch.device] = None) -> dict[str, Any]:

    if update_models:
        model.load_state_dict(torch.load(os.path.join(
            session_dir, "Model State.pckl"), map_location=map_location))
        optimizer.load_state_dict(
            torch.load(os.path.join(session_dir, "Optimizer State.pckl"), map_location=map_location))

    print("Loaded session from", session_dir)

    out_data = {
        'model': model,
        'optimizer': optimizer
    }

    return out_data
