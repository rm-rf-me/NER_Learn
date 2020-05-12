# -*- coding: utf-8 -*-
# @Auther   : liou


def flatten_list(lists):
    flatten_list = []
    for l in lists:
        if isinstance(l, list):
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
