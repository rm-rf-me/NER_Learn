# -*- coding: utf-8 -*-
# @Auther   : liou

import re
import os.path


class CorpusProcess (object):
    def __init__(self):
        self.train_corpus_path = "/home/zyn/PycharmProjects/NER_Learn/data/1980_01rmrb.txt"
        self.preprocess_corpus_path = "/home/zyn/PycharmProjects/NER_Learn/data/pre_process.txt"
        self.tag_corpus_path = "/home/zyn/PycharmProjects/NER_Learn/data/tag_cropus.txt"
        self._map = {u't': u'T', u'nr': 'PER', u'ns': u'ORG', u'nt': u'LOC'}
        self.word_dictionary_path = "/home/zyn/PycharmProjects/NER_Learn/data/word_dictionary.txt"
        self.tag_dictionary_path = "/home/zyn/PycharmProjects/NER_Learn/data/tag_dictionary.txt"

    def read_corpus(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        return lines

    def write_corpus(self, data, file_path):
        f = open(file_path, 'wb')
        f.write(data)
        f.close()

    def q_to_b(self, q_str):
        """全角转换为半角"""
        b_str = ""
        for uchar in q_str:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            b_str += chr(inside_code)
        return b_str

    def process_time(self, words):
        """整合时间序列，把所有连续出现的时间实体标记到一起"""
        pro_words = []
        index = 0
        tmp = u''
        while True:
            word = words[index] if index < len(words) else u''
            if u'/t' in word:
                tmp = tmp.replace(u'/t', u'') + word
            elif tmp:
                pro_words.append(tmp)
                pro_words.append(word)
                tmp = u''
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    def process_nr(self, words):
        """将姓和名字整合到一起"""
        pro_words = []
        index = 0
        while True:
            word = words[index] if index < len(words) else u''
            if u'/nr' in word:
                next_index = index + 1
                if next_index < len(words) and u'nr' in words[next_index]:
                    pro_words.append(
                        word.replace(
                            u'/nr',
                            u'') +
                        words[next_index])
                    index += 1
                else:
                    pro_words.append(word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    def process_long(self, words):
        """处理长的嵌套实体，采用最外层的标记"""
        pro_words = []
        index = 0
        tmp = u''
        while True:
            word = words[index] if index < len(words) else u''
            if u'[' in word:
                tmp += re.sub(pattern=u'/[a-zA-Z]*',
                              repl=u'', string=word.replace(u'[', u''))
            elif u']' in word:
                w = word.split(u']')
                tmp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=w[0])
                pro_words.append(tmp + u'/' + w[1])
                tmp = u''
            elif tmp:
                tmp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    def pre_process(self):
        """进行以上三项预处理并保存"""
        print("pre processing...")
        if os.path.exists(self.preprocess_corpus_path) :
            print ("pre process file is exists")
            return
        lines = self.read_corpus(self.train_corpus_path)
        new_lines = []
        index = 0
        for line in lines:
            index += 1
            if index % 1000 == 0:
                print("%d..." % index, end='')
            words = self.q_to_b(line.strip()).split(u'  ')
            pro_words = self.process_time(words)
            pro_words = self.process_nr(pro_words)
            pro_words = self.process_long(pro_words)
            new_lines.append('  '.join(pro_words[1:]))
        print()
        print("tot %d lines" % index)
        self.write_corpus(
            data='\n'.join(new_lines).encode('utf-8'),
            file_path=self.preprocess_corpus_path)
        print("Done!")

    def pos_to_tag(self, pos):
        tag = self._map.get(pos, None)
        return tag if tag else u'O'

    def BIO_tag(self, tag, index):
        """根据位置选择BIO标签"""
        if index == 0 and tag != u'O':
            return u'B_{}'.format(tag)
        elif tag != u'O':
            return u'I_{}'.format(tag)
        else:
            return tag

    def pos_perform(self, pos):
        """去处先验知识，如nr，np一律标为n"""
        if pos in self._map.keys() and pos != u't':
            return u'n'
        else:
            return pos

    def init_sequence(self, make_vocab = True):
        """初始化词语和标签序列，并保存成特征格式，同时映射并保存词表和标签表"""
        print("init sequence...")
        lines = self.read_corpus(self.preprocess_corpus_path)
        words_list = [line.strip().split('  ')
                      for line in lines if line.strip()]
        del lines
        words_seq = [[word.split(u'/')[0] for word in words]
                     for words in words_list]
        pos_seq = [[word.split(u'/')[1] for word in words]
                   for words in words_list]
        tag_seq = [[self.pos_to_tag(p) for p in pos] for pos in pos_seq]
        self.pos_seq = [[[pos_seq[index][i] for _ in range(len(words_seq[index][i]))]
                         for i in range(len(pos_seq[index]))] for index in range(len(pos_seq))]
        self.tag_seq = [[[self.BIO_tag(tag_seq[index][i], w) for w in range(len(words_seq[index][i]))]
                         for i in range(len(tag_seq[index]))] for index in range(len(tag_seq))]
        self.pos_seq = [[u'un'] +
                        [self.pos_perform(p) for pos in pos_seq for p in pos] +
                        [u'un'] for pos_seq in self.pos_seq]
        self.tag_seq = [[t for tag in tag_seq for t in tag]
                        for tag_seq in self.tag_seq]
        self.word_seq = [[u'<BOS>'] + [w for word in word_seq for w in word] + [u'<BOS>']
                         for word_seq in words_seq]

        if make_vocab:
            self.word2id = self._buile_map(self.word_seq)
            self.tag2id = self._buile_map(self.tag_seq)
        if os.path.exists(self.word_dictionary_path) == 0:
            f = open(self.word_dictionary_path, 'wb')
            for key, val in self.word2id.items() :
                f.write("{} : {}\n".format(key, val).encode())
            f.close()
            print ("word dictionary has benn saved!")
        if os.path.exists(self.tag_dictionary_path) == 0:
            f = open(self.tag_dictionary_path, 'wb')
            for key, val in self.tag2id.items():
                f.write("{} : {}\n".format(key, val).encode())
            f.close()
            print("tag dictionary has benn saved!")


        new_lines = []
        liness = []

        for i in range(len(tag_seq)):
            if i % 1000 == 0:
                print("%d.." % i, end='')
            assert len(self.tag_seq[i]) + 2  == len(self.word_seq[i])

            for j in range(len(self.tag_seq[i])):
                haha = self.word_seq[i][j] + u'  ' + self.tag_seq[i][j]
                new_lines.append(haha)
            new_lines.append(u'\n')
        '\n'.join(new_lines)
        self.write_corpus(
            data='\n'.join(new_lines).encode('utf-8'),
            file_path=self.tag_corpus_path)
        print()
        print("Done!")

    def _buile_map (self, lists) :
        """建立词表"""
        maps = {}
        for list in lists :
            for e in list :
                if e not in maps :
                    maps[e] = len(maps)
        return maps



    def extract_feature(self, word_grams):
        """对每一个窗口抽取特征，此处默认选择五项特征"""
        features, feature_list = [], []
        for i in range(len(word_grams)):
            for j in range(len(word_grams[i])):
                word_gram = word_grams[i][j]
                feature = {u'w-1': word_gram[0],
                           u'w': word_gram[1],
                           u'w+1': word_gram[2],
                           u'w-1/w': word_gram[0] + word_gram[1],
                           u'w/w+1': word_gram[1] + word_gram[2],
                           u'bias': 1.0
                           }
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    def segment_by_window(self, words_list=None, window=3):
        """滑动窗口截取，大小根据特征模板而定"""
        words = []
        begin = 0
        end = window
        for _ in range(1, len(words_list)):
            if end > len(words_list):
                break
            words.append(words_list[begin:end])
            begin += 1
            end += 1
        return words

    def generator(self):
        """生成特征序列和标签"""
        print("Generatoring...")
        word_grams = [self.segment_by_window(
            word_list) for word_list in self.word_seq]
        feature = self.extract_feature(word_grams)
        print("Done!")
        return feature, self.tag_seq


if __name__ == '__main__':
    corpus = CorpusProcess()
    corpus.pre_process()
    corpus.init_sequence()
