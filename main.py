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
    # utils
    'track_time': True,
    'dump_summary': False,
    'export_weights': False
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)

    def train_one_epoch(self, epoch_num):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.config.use_cuda:
                data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
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
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        final_acc = int(correct) / len(data_loader.dataset)
        final_loss = float(loss) / len(data_loader.dataset)
        return final_acc, final_loss

    def train(self):
        self.results = []
        for epoch in range(self.config.epochs):
            print('\n============ train epoch [%d/%d] ============\n' % (epoch+1, self.config.epochs))
            self.train_one_epoch(epoch)
            print('\n============ start testing ==================\n')
            train_acc, train_loss = self.eval(self.train_loader)
            test_acc, test_loss = self.eval(self.test_loader)
            print('Train Accuracy: %.4f\tTrain Loss: %.4f' % (train_acc, train_loss))
            print('Test Accuracy: %.4f\tTest Loss: %.4f' % (test_acc, test_loss))
            curr_res = Dict({
                'epoch': epoch,
                'train_acc': '%.4f' % train_acc,
                'train_loss': '%.4f' % train_loss,
                'test_acc': '%.4f' % test_acc,
                'test_loss': '%.4f' % test_loss,
            })
            self.results.append(curr_res)
            if self.config.dump_summary:
                self.writer.add_scalar("accuracy/train", train_acc, epoch+1)
                self.writer.add_scalar("accuracy/val", test_acc, epoch+1)
                self.writer.add_scalar("loss/train", train_loss, epoch+1)
                self.writer.add_scalar("loss/val", test_loss, epoch+1)
        
        if self.config.track_time:
            exp_time_mins = int(time.time() - self.start_time) // 60
            print('\n[Time]: %.2fmins' % exp_time_mins)


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

if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.train()