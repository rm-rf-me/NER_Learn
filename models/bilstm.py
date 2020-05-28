# -*- coding: utf-8 -*-
# @Auther   : liou

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .utils.bilstm import cal_loss, sort_by_lengths, tensorized

class BiLSTM_base (nn.Module) :
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BiLSTM_base, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths) :
        emb = self.embedding(sents_tensor)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)

        return scores

    def test (self, sents_tensor, lengths, _) :
        logits = self.forward(sents_tensor, lengths)
        _, batch_tagids = torch.max (logits, dim = 2)
        return batch_tagids

class BILSTM_Model (object) :
    def __init__(self, vocab_size, out_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = 128
        self.hidden_size = 128
        self.model = BiLSTM_base(vocab_size, self.emb_size, self.hidden_size, out_size).to(self.device)
        self.loss_fun = cal_loss
        self.epoches = 30
        self.print_step = 5
        self.lr = 0.001
        self.batch_size = 64
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train_step (self, batch_sents, batch_tags, word2id, tag2id) :
        self.model.train()
        self.step += 1
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        targets, lengths = tensorized(batch_tags, tag2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets = targets.to(self.device)

        scores = self.model(tensorized_sents, lengths)

        self.optimizer.zero_grad()
        loss = self.loss_fun (scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train (self, word_lists, tag_lists,
               dev_word_lists, dev_tag_lists,
               word2id, tag2id) :
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range (1, self.epoches + 1):
            self.step = 0
            losses = 0.
            for ind in range (0, len (word_lists), B) :
                batch_sents = word_lists[ind : ind+B]
                batch_tags = tag_lists[ind: ind+B]
                losses += self.train_step (batch_sents, batch_tags, word2id, tag2id)

                if self.step % self.print_step == 0 :
                    total_step = (len (word_lists) // B + 1)
                    print ("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step, 100. * self.step / total_step, losses / self.print_step
                    ))
                    losses = 0.
            val_loss = self.validate(dev_word_lists, dev_tag_lists, word2id, tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))


    def validate (self, dev_word_lists, dev_tag_lists, word2id, tag2id) :
        self.model.eval ()
        with torch.no_grad() :
            val_losses = 0.
            val_step = 0
            for ind in range (0, len(dev_word_lists), self.batch_size):
                val_step += 1
                batch_sents = dev_word_lists[ind : ind+self.batch_size]
                batch_tags = dev_tag_lists[ind : ind+self.batch_size]
                tensorized_sents, lengths = tensorized(batch_sents, word2id)
                targets, lengths = tensorized(batch_tags, tag2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets = targets.to(self.device)

                scores = self.model (tensorized_sents, lengths)

                loss = self.loss_fun (scores, targets, tag2id).to(self.device)

                val_losses += loss.item()

            val_losses = val_losses / val_step
            if val_losses < self._best_val_loss :
                print("保存模型。。。")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_losses
            return val_losses

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            print ("testing...")
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)


        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            for j in range(lengths[i]):
                tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists
