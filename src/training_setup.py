import math
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.random
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import utils

def get_cosine_scheduler_with_warmup_and_restart(optimizer, epochs, steps_per_epoch):
    """
    带预热和重启的余弦退火学习率调度器
    
    参数:
        optimizer: 优化器
        epochs: 总训练轮数
        steps_per_epoch: 每轮的步数
    """
    # 总步数
    total_steps = epochs * steps_per_epoch
    
    # 预热步数 (总步数的5%)
    warmup_steps = int(0.05 * total_steps)
    
    # 第一个重启周期长度 (总步数的1/3)
    first_cycle = int(total_steps / 3)
    
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 预热后的退火重启阶段
        current_step = current_step - warmup_steps
        
        # 计算当前所在的周期
        if first_cycle == 0:
            return 0.1  # 防止除零错误
            
        # 找出当前所在的周期和周期内的步数
        cycle = 1
        current_cycle_steps = current_step
        
        # 每个新周期的长度是前一个周期的2倍
        cycle_length = first_cycle
        while current_cycle_steps >= cycle_length:
            current_cycle_steps -= cycle_length
            cycle_length *= 2  # 周期长度翻倍
            cycle += 1
            
            # 防止无限循环
            if cycle > 10:  
                break
        
        # 在当前周期内计算余弦衰减
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_cycle_steps / cycle_length))
        
        # 根据周期数调整衰减范围
        decay_factor = 0.9 ** (cycle - 1)
        
        return max(0.1, cosine_decay * decay_factor)  # 确保学习率不低于初始值的10%
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# def cli(
#         scan_hparams: Dict[str, Any],
#         prod_hparams: Dict[str, Any]
# ) -> Tuple[Dict[str, Any], argparse.Namespace, Dict[str, Any]]:

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--prod', action='store_true', default=False)
#     args = parser.parse_args()

#     if args.prod:
#         print('+++ This is a *prod* run +++')
#         setup_args = {
#             'scheduler': get_cosine_scheduler_with_warmup_and_restart,
#             'n_epochs': 10000,
#             'save_model': True
#         }
#         hparams = prod_hparams
#     else:
#         print('+++ This is a scanning run +++')
#         setup_args = {
#             'save_model': True,
#             'n_epochs': 10000,
#             'scheduler': get_cosine_scheduler_with_warmup_and_restart
#         }
#         hparams = scan_hparams
#     return setup_args, args, hparams

def cli(
        scan_hparams: Dict[str, Any],
        prod_hparams: Dict[str, Any]
) -> Tuple[Dict[str, Any], argparse.Namespace, Dict[str, Any]]:

    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true', default=False)
    args = parser.parse_args()

    if not args.prod:
        print('+++ This is a *prod* run +++')
        setup_args = {
            'scheduler': None,
            'n_epochs': 10000,
            'save_model': True
        }
        hparams = prod_hparams
    else:
        print('+++ This is a scanning run +++')
        setup_args = {
            'save_model': True,
            'n_epochs': 10000,
            'scheduler': None
        }
        hparams = scan_hparams
    return setup_args, args, hparams


def clear_device():
    torch.cuda.empty_cache()


def setup_device() -> torch.DeviceObjType:
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    return device

# def setup_device() -> torch.DeviceObjType:
#     if torch.cuda.is_available():
#         # 对于DataParallel，使用第一张GPU作为主GPU
#         dev = 'cuda:0'
#         print(f"主GPU: {torch.cuda.get_device_name(0)}")
        
#         # 打印出所有可用的GPU信息
#         if torch.cuda.device_count() > 1:
#             print(f"将使用以下 {torch.cuda.device_count()} 张GPU:")
#             for i in range(torch.cuda.device_count()):
#                 print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
#     else:
#         dev = 'cpu'
#         print("使用CPU训练")
#     device = torch.device(dev)
#     return device


def split_dataset(dataset: torch.utils.data.Dataset) -> torch.utils.data.Subset:
    train_size = int(len(dataset) * 0.8)
    test_size = int((len(dataset) - train_size) * 0.5)
    valid_size = len(dataset) - (train_size + test_size)
    return torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])


import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super(CosineSimilarityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x1, x2):

        cosine_sim = F.cosine_similarity(x1, x2, dim=1)

        mse_loss = F.mse_loss(x1, x2)

        loss = (1 - self.alpha) * (1-cosine_sim.mean()) + mse_loss * self.alpha

        return loss
    
class EnhancedMolecularLoss(nn.Module):
    def __init__(self, cosine_weight=0.3, binary_weight=0.4, mse_weight=0.3):
        super(EnhancedMolecularLoss, self).__init__()
        self.cosine_weight = cosine_weight
        self.binary_weight = binary_weight
        self.mse_weight = mse_weight
        
        # 特征区间
        self.morgan_start = 0
        self.morgan_end = 1024
        self.torsion_start = self.morgan_end
        self.torsion_end = self.torsion_start + 857
        self.maccs_start = self.torsion_end
        self.maccs_end = self.maccs_start + 167
        # 碎片和加合物特征从maccs_end开始
        
    def forward(self, x1, x2):
        # 余弦相似度损失
        cosine_sim = F.cosine_similarity(x1, x2, dim=1)
        cosine_loss = 1.0 - cosine_sim.mean()
        
        # 二元特征损失 (Morgan, Torsion, MACCS)
        binary_part_x1 = x1[:, :self.maccs_end]
        binary_part_x2 = x2[:, :self.maccs_end]
        binary_loss = F.binary_cross_entropy(binary_part_x1, binary_part_x2)
        
        # 连续特征损失 (碎片水平和加合物)
        continuous_part_x1 = x1[:, self.maccs_end:]
        continuous_part_x2 = x2[:, self.maccs_end:]
        mse_loss = F.mse_loss(continuous_part_x1, continuous_part_x2)
        
        # 总损失
        total_loss = (
            self.cosine_weight * cosine_loss + 
            self.binary_weight * binary_loss +
            self.mse_weight * mse_loss
        )
        
        return total_loss

class TrainingSetup:
    def __init__(
            self,
            model: torch.nn.Module,
            dataset: torch.utils.data.Dataset,
            outdir: str,
            n_epochs: int = 100,
            lr: float = 3e-4,
            batch_size: int = 300,
            optimizer: optim.Optimizer = optim.Adam,
            dataloader=DataLoader,
            device: Optional[torch.DeviceObjType] = None,
            scheduler: optim.lr_scheduler._LRScheduler = None,
            save_model: bool = False,
            checkpoint: Optional[Any] = None
    ):
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.save_model = save_model
        self.scheduler = scheduler
        self.train_data, self.test_data, self.validation_data = split_dataset(dataset)
        self.checkpoint = None if checkpoint is None else torch.load(checkpoint)

        self.train_loader = dataloader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = dataloader(self.test_data, batch_size=self.batch_size)
        self.device = setup_device() if device is None else device
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.optimizer = optimizer

    def _train_evaluate(self):
        torch.manual_seed(utils.RANDOM_SEED)
        self.model = self.model.to(self.device)
        criterion = CosineSimilarityLoss()
        # criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        if self.scheduler is not None:
            scheduler = self.scheduler(
                optimizer,
                epochs=self.n_epochs,
                steps_per_epoch=len(self.train_loader)
            )
        else:
            scheduler = None

        if self.checkpoint is not None:
            print('Loading checkpoint')
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        losses = {'train': {}, 'test': {}}
        state_to_save = {'loss': np.inf}
        best_loss = 10000
        test_loss = 0
        best_model_path = "None"
        for epoch in range(self.n_epochs):
            losses['train'][epoch] = 0
            losses['test'][epoch] = 0

            for inputs1, targets in self.train_loader:

                optimizer.zero_grad()

                loss = criterion(
                    self.model(inputs1.to(self.device)),
                    targets.to(self.device)
                )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                losses['train'][epoch] += loss.item() / len(self.train_loader)

            for inputs1, targets in self.test_loader:
                with torch.no_grad():

                    loss = criterion(
                        self.model(inputs1.to(self.device)),
                        targets.float().to(self.device)
                    )
                    losses['test'][epoch] += loss.item() / len(self.test_loader)

                    if True:  
                        state_to_save['epoch'] = epoch
                        state_to_save['model_state_dict'] = self.model.state_dict()
                        state_to_save['model_kwargs'] = self.model.kwargs
                        if self.save_model:
                            state_to_save['optimizer_state_dict'] = optimizer.state_dict()
                        state_to_save['loss'] = losses['test'][epoch]

            if losses['test'][epoch] < best_loss:

                if best_model_path != "None":
                    print("remove:", f'{self.outdir}/{best_model_path}')
                    os.remove(f'{self.outdir}/{best_model_path}')
                best_model_path = f'{epoch}_best_checkpoint.pt'
                print("save:", f'{epoch}_best_checkpoint.pt')
                best_loss = losses['test'][epoch]
                torch.save(state_to_save, f'{self.outdir}/{epoch}_best_checkpoint.pt')
            print(f'Epoch: {epoch}: Loss train: [{losses["train"][epoch]}] | test: [{losses["test"][epoch]}]')
        if self.save_model:

            torch.save(state_to_save, f'{self.outdir}/best_checkpoint.pt')
        return losses

    def _plot_curves(self, curves: Dict[str, np.array]):
        plt.figure()
        for name, loss in curves.items():
            plt.plot(list(loss.values()), label=f'{name}')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        plt.savefig(f'{self.outdir}/learning_curves.png')

    def _save_stats(self, curves: Dict[str, np.array]):
        pd.DataFrame(curves).to_csv(f'{self.outdir}/learning_curves.tsv')

    def train(self):
        clear_device()
        self.curves = self._train_evaluate()
        self._plot_curves(self.curves)
        self._save_stats(self.curves)

# class TrainingSetup:
#     def __init__(
#             self,
#             model: torch.nn.Module,
#             dataset: torch.utils.data.Dataset,
#             outdir: str,
#             n_epochs: int = 100,
#             lr: float = 3e-4,
#             batch_size: int = 300,
#             optimizer: optim.Optimizer = optim.Adam,
#             dataloader=DataLoader,
#             device: Optional[torch.DeviceObjType] = None,
#             scheduler: optim.lr_scheduler._LRScheduler = None,
#             save_model: bool = False,
#             checkpoint: Optional[Any] = None
#     ):
#         self.model = model
#         self.batch_size = batch_size
#         self.lr = lr
#         self.n_epochs = n_epochs
#         self.save_model = save_model
#         self.scheduler = scheduler
#         self.train_data, self.test_data, self.validation_data = split_dataset(dataset)
#         self.checkpoint = None if checkpoint is None else torch.load(checkpoint)

#         self.train_loader = dataloader(self.train_data, batch_size=self.batch_size, shuffle=True)
#         self.test_loader = dataloader(self.test_data, batch_size=self.batch_size)
#         self.device = setup_device() if device is None else device
#         self.outdir = outdir
#         os.makedirs(self.outdir, exist_ok=True)
#         self.optimizer = optimizer

#     def _train_evaluate(self):
#         torch.manual_seed(utils.RANDOM_SEED)
#         self.model = self.model.to(self.device)
#         # criterion = CosineSimilarityLoss()
#         # criterion = nn.MSELoss()
#         criterion = EnhancedMolecularLoss()
#         optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
#         if self.scheduler is not None:
#             scheduler = self.scheduler(
#                 optimizer,
#                 epochs=self.n_epochs,
#                 steps_per_epoch=len(self.train_loader)
#             )
#         else:
#             scheduler = None

#         if self.checkpoint is not None:
#             print('Loading checkpoint')
#             self.model.load_state_dict(self.checkpoint['model_state_dict'])
#             optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

#         losses = {'train': {}, 'test': {}}
#         state_to_save = {'loss': np.inf}
#         best_loss = 10000
#         test_loss = 0
#         best_model_path = "None"
#         for epoch in range(self.n_epochs):
#             losses['train'][epoch] = 0
#             losses['test'][epoch] = 0

#             for inputs1, targets in self.train_loader:

#                 optimizer.zero_grad()

#                 loss = criterion(
#                     self.model(inputs1.to(self.device)),
#                     targets.to(self.device)
#                 )
#                 loss.backward()
#                 optimizer.step()
#                 if scheduler is not None:
#                     scheduler.step()

#                 losses['train'][epoch] += loss.item() / len(self.train_loader)

#             for inputs1, targets in self.test_loader:
#                 with torch.no_grad():

#                     loss = criterion(
#                         self.model(inputs1.to(self.device)),
#                         targets.float().to(self.device)
#                     )
#                     losses['test'][epoch] += loss.item() / len(self.test_loader)

#                     if True:  
#                         state_to_save['epoch'] = epoch
#                         state_to_save['model_state_dict'] = self.model.state_dict()
#                         state_to_save['model_kwargs'] = self.model.kwargs
#                         if self.save_model:
#                             state_to_save['optimizer_state_dict'] = optimizer.state_dict()
#                         state_to_save['loss'] = losses['test'][epoch]

#             if losses['test'][epoch] < best_loss:

#                 if best_model_path != "None":
#                     print("remove:", f'{self.outdir}/{best_model_path}')
#                     os.remove(f'{self.outdir}/{best_model_path}')
#                 best_model_path = f'{epoch}_best_checkpoint.pt'
#                 print("save:", f'{epoch}_best_checkpoint.pt')
#                 best_loss = losses['test'][epoch]
#                 torch.save(state_to_save, f'{self.outdir}/{epoch}_best_checkpoint.pt')
#             print(f'Epoch: {epoch}: Loss train: [{losses["train"][epoch]}] | test: [{losses["test"][epoch]}]')
#         if self.save_model:

#             torch.save(state_to_save, f'{self.outdir}/best_checkpoint.pt')
#         return losses

#     def _plot_curves(self, curves: Dict[str, np.array]):
#         plt.figure()
#         for name, loss in curves.items():
#             plt.plot(list(loss.values()), label=f'{name}')
#             plt.yscale('log')
#             plt.legend()
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#         plt.savefig(f'{self.outdir}/learning_curves.png')

#     def _save_stats(self, curves: Dict[str, np.array]):
#         pd.DataFrame(curves).to_csv(f'{self.outdir}/learning_curves.tsv')

#     def train(self):
#         clear_device()
#         self.curves = self._train_evaluate()
#         self._plot_curves(self.curves)
#         self._save_stats(self.curves)
