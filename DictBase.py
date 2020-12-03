# -*- coding: utf-8 -*-

'''
@Time    : 2020/12/3 下午4:38
@Author  : liou
@FileName: DictBase.py
@Software: PyCharm
 
'''

from models.DictBase import Dict_Base


def main ():

    dict_path = 'data/pos_data/ci/pos_word_dictionary.txt'
    dic = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()[0]
            dic.append(word)

    dict_base_model = Dict_Base(dic)

    res = dict_base_model.predict('迈向充满希望的新世纪——一九九八年新年讲话')

    print (res)

if __name__ == '__main__':
    main()
