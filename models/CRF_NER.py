# -*- coding: utf-8 -*-
# @Auther   : liou

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from .utils.CorpusProcess import CorpusProcess
from .utils.Metrics import Metrics


class CRF_NER (object):
    def __init__(self, algorithm="lbfgs",
                 c1="0.1", c2="0.2",
                 max_iter=100, model_path="/home/zyn/PycharmProjects/NLPLearn/save/model.pkl",
                 remove_O=True):
        print ("init model...")
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iter
        self.model_path = model_path
        self.remove_O = remove_O
        self.model = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True)
        print ("Done!")

    def train_and_evl(self, x, y):
        print ("training...")
        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        self.model.fit(x_train, y_train)
        print ("Done!\npredicting...")
        labels = list(self.model.classes_)
        y_predict = self.model.predict(x_test)
        metrics = Metrics(y_test, y_predict, remove_O=self.remove_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()
        self.save_model()
        print ("model has been saved")


    def predict(self, sentence):
        self.load_model()
        u_sent = self.corpus.q_to_b(sentence)
        word_lists = [[u'<BOS>'] + [c for c in u_sent] + [u'<EOS>']]
        word_gram = [self.corpus.segment_by_window(
            word_list) for word_list in word_lists]
        feature = self.corpus.extract_feature(word_gram)
        y_predict = self.model.predict(feature)
        entity = u''
        for index in range(len(y_predict)):
            if y_predict[0][index] != u'O':
                if index > 0 and y_predict[0][index][-1] != y_predict[0][index - 1][-1]:
                    entity += u' '
                entity += u_sent[index]
            elif entity[-1] != u' ':
                entity += u' '
        return entity

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)

