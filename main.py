# -*- coding: utf-8 -*-
# @Auther   : liou
from models.CRF_NER import CRF_NER
from models.utils.CorpusProcess import CorpusProcess
from models.utils.Metrics import Metrics
from models.hmm import HMM
from data import build_corpus

def main () :

    print ("loading data...")
    data = CorpusProcess()
    data.pre_process()
    data.init_sequence()

    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    # dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    print ("traing and testing hmm...")

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)

    # 评估hmm模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=True)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    print ("training and testing crf...")
    x, y = data.generator()
    ner = CRF_NER()
    model = ner.train_and_evl(x, y)

if __name__ == '__main__':
    main ()
