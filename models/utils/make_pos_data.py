# -*- coding: utf-8 -*-
# @Auther   : liou

import re
import os.path


def read_corpus(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    return lines

def _buile_map (lists) :
    """建立词表"""
    maps = {}
    for list in lists :
        for e in list :
            if e not in maps :
                maps[e] = len(maps)
    return maps

pre_path = "/Users/liou/PycharmProjects/NER_Learn-master"
preprocess_corpus_path = pre_path + "/data/pre_process.txt"
word_dictionary_path = pre_path + "/data/pos_data/ci/pos_word_dictionary.txt"
pos_dictionary_path = pre_path + "/data/pos_data/ci/pos_dictionary.txt"
tag_corpus_path = pre_path + "/data/pos_data/ci/pos_cropus.txt"

lines = read_corpus(preprocess_corpus_path)

words_list = [line.strip().split('  ')
              for line in lines if line.strip()]
del lines

words_seq = [[word.split(u'/')[0]
              for word in words]
             for words in words_list]

pos_seq = [[word.split(u'/')[1]
            for word in words]
           for words in words_list]


pos_seq = [[u'un'] + [pos for pos in pos_seq] + [u'un'] for pos_seq in pos_seq]

word_seq = [[u'<BOS>'] + [word for word in word_seq] + [u'<BOS>'] for word_seq in words_seq]

word2id = _buile_map(word_seq)
tag2id = _buile_map(pos_seq)

f = open(word_dictionary_path, 'wb')
for key, val in word2id.items():
    f.write("{} : {}\n".format(key, val).encode())
f.close()
print("word dictionary has benn saved!")

f = open(pos_dictionary_path, 'wb')
for key, val in tag2id.items():
    f.write("{} : {}\n".format(key, val).encode())
f.close()
print("tag dictionary has benn saved!")

new_lines = []

for i in range(len(pos_seq)):
    if i % 1000 == 0:
        print("%d.." % i, end='')
    assert len(pos_seq[i])  == len(word_seq[i])

    for j in range(len(pos_seq[i])):
        haha = word_seq[i][j] + u'  ' + pos_seq[i][j]
        new_lines.append(haha)
    new_lines.append(u'\n')
'\n'.join(new_lines)


ff = open (tag_corpus_path, 'wb')
ff.write ('\n'.join(new_lines).encode('utf-8'))
ff.close()

