# -*- coding: utf-8 -*-
# @Auther   : liou

import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_loss(logits, targets, tag2id):
    PAD = tag2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss

def sort_by_lengths (word_lists, tag_lists) :
    pairs = list(zip (word_lists, tag_lists))
    indices = sorted (range (len (pairs)), key=lambda k: len(pairs[k][0]), reverse=True)
    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list (zip(*pairs))

    return word_lists, tag_lists, indices

def tensorized (batch, maps) :
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len (batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l, in enumerate(batch) :
        for j, e in enumerate(l) :
            batch_tensor[i][j] = maps.get(e, UNK)
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths

def extend_maps (word2id, tag2id) :
    word2id['<unk>'] = len (word2id)
    word2id['<pad>'] = len (word2id)
    tag2id['<unk>'] = len (tag2id)
    tag2id['<pad>'] = len (tag2id)

    return word2id, tag2id