# -*- coding: utf-8 -*-
# @Auther   : liou

from models.CRF_NER import CRF_NER
from models.utils.CorpusProcess import CorpusProcess
from models.utils.Metrics import Metrics
from models.hmm import HMM
from data import build_corpus
from models.utils.bilstm import extend_maps
import time
from models.bilstm import BILSTM_Model
from models.utils.util import save_model, load_model
import torch

def main () :
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id)

    start = time.time()
    vocab_size = len (word2id)
    out_size = len (tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists,
                       word2id, tag2id)
    model_name = "bilstm"
    save_model(bilstm_model, "./save/"+model_name+".pkl")

    # print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    # print("评估{}模型中...".format(model_name))



    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    print ("cal the res...")
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=False)
    metrics.haha()
    #metrics.report_confusion_matrix()

if __name__ == '__main__':
    main ()
