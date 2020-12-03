# -*- coding: utf-8 -*-
# @Auther   : liou

import re
import os.path
import json

pre_path = "/Users/liou/PycharmProjects/NER_Learn-master"
preprocess_corpus_path = pre_path + "/data/pre_process.txt"
word_dictionary_path = pre_path + "/data/pos_data/ci/pos_word_dictionary.txt"
pos_dictionary_path = pre_path + "/data/pos_data/ci/pos_dictionary.txt"
tag_corpus_path = pre_path + "/data/pos_data/ci/pos_cropus.txt"
pos_json_path = pre_path + "/data/pos_data/ci/pos_word.json"
pos_trans_json = pre_path + "/data/pos_data/ci/pos_trans.json"
pos_freq = pre_path + "/data/pos_data/ci/pos_freq.json"

def read_corpus(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    return lines

words = read_corpus(pos_dictionary_path)

pos_corpus = read_corpus(tag_corpus_path)

haha = {}
freq = {}

i = 0

for line in pos_corpus :
    if line == '\n':
        continue
    i += 1;
    if i % 10000 == 0:
        print (i)
    word = line.split('  ')[0]
    pos = line.split('  ')[1]

    if freq.get (pos) is None:
        freq[pos] = 1
    else :
        freq[pos] += 1

    if haha.get(word) is None:
        tmp = {}
        tmp[pos] = 1
        haha[word] = tmp
    else :
        if haha[word].get(pos) is None :
            haha[word][pos] = 1
        else:
            haha[word][pos] += 1

trans = {}

pos_list = [line.strip().split('  ')[1] for line in pos_corpus if line.strip()]

now_pos = ''
next_pos = pos_list[0]

for i in range (len (pos_list) - 1):
    now_pos = next_pos
    next_pos = pos_list[i + 1]
    tran = now_pos + ' ' + next_pos
    if trans.get (tran) is None :
        trans[tran] = 1
    else :
        trans[tran] += 1

with open(pos_json_path,'w') as file_obj:
     json.dump(haha,file_obj,ensure_ascii=False)

with open(pos_trans_json, 'w') as file_obj:
    json.dump(trans, file_obj, ensure_ascii=False)

with open(pos_freq, 'w') as file_obj:
    json.dump(freq, file_obj, ensure_ascii=False)




