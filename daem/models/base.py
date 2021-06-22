import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Precision, Recall, F1

class EntityMatchingModule(nn.Module):
    def __init__(self):
        super(EntityMatchingModule, self).__init__()

        self.train_prec = Precision(ignore_index=0)
        self.train_rec = Recall(ignore_index=0)
        self.train_f1 = F1(ignore_index=0)

        self.valid_prec = Precision(ignore_index=0)
        self.valid_rec = Recall(ignore_index=0)
        self.valid_f1 = F1(ignore_index=0)

        self.test_prec = Precision(ignore_index=0)
        self.test_rec = Recall(ignore_index=0)
        self.test_f1 = F1(ignore_index=0)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        p = self(x)
        loss = F.cross_entropy(p, y)
        self.log('train_loss', loss)
        p = torch.softmax(p, 1)
        self.train_prec(p, y)
        self.train_rec(p, y)
        self.train_f1(p, y)
        return loss

    def training_epoch_end(self, outs):
        self.log('tr_pr', self.train_prec.compute(), prog_bar=True)
        self.log('tr_r', self.train_rec.compute(), prog_bar=True)
        self.log('tr_f', self.train_f1.compute(), prog_bar=True)
        self.train_prec.reset()
        self.train_rec.reset()
        self.train_f1.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        p = self(x)
        loss = F.cross_entropy(p, y)
        p = torch.softmax(p, 1)
        self.valid_prec(p, y)
        self.valid_rec(p, y)
        self.valid_f1(p, y)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outs):
        self.log('vl_p', self.valid_prec.compute(), prog_bar=True)
        self.log('vl_r', self.valid_rec.compute(), prog_bar=True)
        self.log('vl_f', self.valid_f1.compute(), prog_bar=True)
        self.valid_prec.reset()
        self.valid_rec.reset()
        self.valid_f1.reset()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        p = self(x)
        loss = F.cross_entropy(p, y)
        p = torch.softmax(p, 1)
        self.test_prec(p, y)
        self.test_rec(p, y)
        self.test_f1(p, y)
        self.log('test_loss', loss)

    def test_epoch_end(self, outs):
        self.log('test_prec_epoch', self.test_prec.compute())
        self.log('test_rec_epoch', self.test_rec.compute())
        self.log('test_f1_epoch', self.test_f1.compute())