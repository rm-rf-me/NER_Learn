# -*- coding: utf-8 -*-

'''
@Time    : 2020/12/3 上午10:53
@Author  : liou
@FileName: DictBase.py
@Software: PyCharm
 
'''

import torch


class Dict_Base (object) :
    def __init__(self, word_dict):
        self.word_dict = word_dict
        self.max_len = max(map(len,[w for w in self.word_dict]))

    def predict (self, seq, method='both'):
        if method == 'both':
            print('fmm: ', '/'.join(self._fmm(seq)))
            print('rmm: ', '/'.join(self._rmm(seq)))
            print('bimm: ', '/'.join(self._bimm(seq)))
        elif method == 'fmm':
            print('fmm: ', '/'.join(self._fmm(seq)))

        elif method == 'rmm':
            print('rmm: ', '/'.join(self._rmm(seq)))

        elif method == 'bimm':
            print('bimm: ', '/'.join(self._bimm(seq)))





    def _fmm(self, seq):
        pre= []
        index = 0
        text_size = len(seq)
        while text_size > index:
            for size in range(self.max_len + index, index, -1):
                piece = seq[index:size]
                if piece in self.word_dict:
                    index = size - 1
                    break
            index = index + 1
            pre.append(piece)
        return pre

    def _rmm(self, seq):
        pre= []
        index = len(seq)
        window_size = min(index, self.max_len)
        while index > 0:
            for size in range(index-window_size, index):
                piece = seq[size:index]
                if piece in self.word_dict:
                    index = size + 1
                    break
            index = index - 1
            pre.append(piece)
        pre.reverse()
        return pre

    def _bimm(self, seq):
        res_fmm = self.fmm(seq)
        res_rmm = self.rmm(seq)
        if len(res_fmm) == len(res_rmm):
            if res_fmm == res_rmm:
                return res_fmm
            else:
                f_word_count = len([w for w in res_fmm if len(w) == 1])
                r_word_count = len([w for w in res_rmm if len(w) == 1])
                return res_fmm if f_word_count < r_word_count else res_rmm
        else:
            return res_fmm if len(res_fmm) < len(res_rmm) else res_rmm
