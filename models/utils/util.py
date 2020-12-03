# -*- coding: utf-8 -*-
# @Auther   : liou

import pickle

def flatten_list(lists):
    flatten_list = []
    for l in lists:
        if isinstance(l, list):
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

def save_model (model, file_name) :
    with open (file_name, 'wb') as f:
        pickle.dump(model, f)

def load_model (file_name) :
    with open (file_name, 'rb') as f:
        model = pickle.load(f)
    return model