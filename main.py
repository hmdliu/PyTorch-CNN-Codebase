#########################################################################
# A compact PyTorch codebase for CNN experiments
# Created by: Hammond Liu (hammond.liu@nyu.edu)
# License: GPL-3.0
# Project Url: https://github.com/hmdliu/PyTorch-CNN-Codebase/
#########################################################################

import os
import time
from addict import Dict
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from util.model import Sample_Net
from util.dataset import get_mnist, get_cifar10
from util.acuuracy import batch_classification_accuracy

PATH = os.getcwd()
CONFIG = Dict({
    # model
    'model': 'sample',
    'model_args': {'in_feats': 784, 'out_feats': 10},
    # dataset
    'dataset': 'mnist',
    'train_batch_size': 64,
    'test_batch_size': 64,
    'dl_workers': 2,
    # training setting
    'seed': 1,
    'lr': 0.01,
    'epochs': 3,
    'step': 200,
    'optimizer': 'sgd',
    'criterion': 'cross_entropy',
    'accr_type': 'classification',
    # utils
    'track_time': True,
    'dump_summary': True,
    'export_weights': True
})

class Trainer():
    def __init__(self, config):
        self.config = config
        self.init_trainer()
        self.init_data_loader()
        self.init_model()
        
    def init_trainer(self):
        torch.manual_seed(self.config.seed)
        self.config.use_cuda = (self.config.use_cuda and torch.cuda.is_available())
        if self.config.track_time:
            self.start_time = time.time()
        if self.config.dump_summary:
            smy_path = os.path.join(PATH, 'results')
            if not os.path.isdir(smy_path):
                os.makedirs(smy_path)
            self.writer = SummaryWriter(smy_path)

    def init_data_loader(self):
        train_dataset = get_dataset(self.config.dataset, mode='train')
        test_dataset = get_dataset(self.config.dataset, mode='val')
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dl_workers
        )
        self.test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=self.config.dl_workers
        )
    
    def init_model(self):
        model = get_model(self.config.model, self.config.model_args)
        self.device = torch.device("cuda:0" if self.config.use_cuda else "cpu")
        if self.config.use_cuda and torch.cuda.device_count() > 1:
            device_ids = [i for i in range(torch.cuda.device_count())]
            model = nn.DataParallel(model, device_ids)
        self.model = model.to(self.device)
        self.optimizer = get_optimizer(self.config.optimizer, self.model, self.config.lr)
        self.criterion = get_criterion(self.config.criterion)
        self.calc_accr = get_calc_accr(self.config.accr_type)

    def train_one_epoch(self, epoch_num):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.config.use_cuda:
                data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.step == 0:
                print('Step %d: [%d/%d], Loss: %.4f' % (
                    batch_idx,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    loss.item()
                ))

    def eval(self, data_loader):
        self.model.eval()
        loss, correct = 0, 0
        for data, target in data_loader:
            if self.config.use_cuda:
                data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            batch_correct, batch_loss = self.calc_accr(output, target)
            correct += batch_correct
            loss += batch_loss
        final_acc = int(correct) / len(data_loader.dataset)
        final_loss = float(loss) / len(data_loader.dataset)
        return final_acc, final_loss

    def train(self):
        self.best_pred = Dict({'accuracy': 0.0, 'state_dict': self.model.state_dict()})
        for epoch in range(self.config.epochs):
            print('\n============ train epoch [%d/%d] ============\n' % (epoch+1, self.config.epochs))
            self.train_one_epoch(epoch)
            print('\n============ start testing ==================\n')
            train_acc, train_loss = self.eval(self.train_loader)
            test_acc, test_loss = self.eval(self.test_loader)
            print('Train Accuracy: %.4f\tTrain Loss: %.4f' % (train_acc, train_loss))
            print('Val Accuracy: %.4f\tVal Loss: %.4f' % (test_acc, test_loss))
            if test_acc > self.best_pred.accuracy:
                self.best_pred.accuracy = test_acc
                self.best_pred.state_dict = self.model.state_dict()
            if self.config.dump_summary:
                self.writer.add_scalar("accuracy/train", train_acc, epoch+1)
                self.writer.add_scalar("accuracy/val", test_acc, epoch+1)
                self.writer.add_scalar("loss/train", train_loss, epoch+1)
                self.writer.add_scalar("loss/val", test_loss, epoch+1)
        
        print('\n[Best Pred]: %.4f' % self.best_pred.accuracy)

        if self.config.track_time:
            exp_time_mins = int(time.time() - self.start_time) // 60
            print('\n[Time]: %.2fmins\n' % exp_time_mins)
        
        if self.config.export_weights:
            export_dir = os.path.join(PATH, 'results')
            export_pre = os.path.join(export_dir, '%s_%s_%s' % (
                self.config.model, 
                self.config.dataset, 
                int(time.time())
            ))
            if not os.path.isdir(export_dir):
                os.makedirs(export_dir)
            with open(export_pre + '.txt', 'w') as f:
                info = []
                for k, v in self.config.items():
                    info.append('%s: %s' % (k, v))
                info.append('\n[Best Pred]: %.4f\n' % self.best_pred.accuracy)
                info.append(str(self.model))
                f.write('\n'.join(info))
            torch.save(self.best_pred.state_dict, export_pre + '.pth')
            print('[Best Weights]: Successfully exported.\n')


def get_dataset(dataset, mode='train'):
    avail_datasets = {
        'mnist': get_mnist,
        'cifar10': get_cifar10
    }
    assert dataset in avail_datasets
    assert mode in ('train', 'val')
    return avail_datasets[dataset](mode == 'train')

def get_model(model, model_args):
    avail_models = {
        'sample': Sample_Net
    }
    assert model in avail_models
    return avail_models[model](**model_args)

def get_optimizer(optimizer, model, lr):
    avail_optimizer = {
        'sgd': optim.SGD
    }
    assert optimizer in avail_optimizer
    return avail_optimizer[optimizer](model.parameters(), lr=lr)

def get_criterion(criterion):
    avail_criterion = {
        'cross_entropy': F.cross_entropy
    }
    assert criterion in avail_criterion
    return avail_criterion[criterion]

def get_calc_accr(acc_type):
    avail_acc_type = {
        'classification': batch_classification_accuracy
    }
    assert acc_type in avail_acc_type
    return avail_acc_type[acc_type]

if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.train()