import os
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.externals import joblib
from model.dataset.dbd_reader import DbdReader


class Detector():

    def __init__(self, session, model_interface, model_dir="", name="", ignore_t=False, load_if_exist=True):
        self.model_interface = model_interface
        self.model_interface.build(session, predict=False, projection=False)
        self.classifier = SVC(kernel="rbf", probability=True)
        self.ignore_t = ignore_t
        self.model_dir = model_dir
        self.name = name if name else self.__class__.__name__.lower()
        if load_if_exist and os.path.isfile(self._make_model_path()):
            self.load()
    
    def save(self):
        if not self.model_dir:
            raise Exception("Detector's model path is not specified.")
        elif not os.path.isdir(self.model_dir):
            raise Exception("No such directory")
        else:
            joblib.dump(self.classifier, self._make_model_path())
    
    def load(self):
        self.classifier = joblib.load(self._make_model_path())

    def _make_model_path(self):
        return os.path.join(self.model_dir, "{0}.pkl".format(self.name))

    def train(self, session, utterance_pairs, labels, check_interval=-1, max_iteration=-1):
        X = []
        t = [lb.get_label(self.ignore_t) for lb in labels]
        for pair in utterance_pairs:
            x = self.extract_feature(session, pair)
            X.append(x)

        self.classifier.fit(X, t)
    
    def predict(self, session, utterance_pair, threshold=0.45):
        x = self.extract_feature(session, utterance_pair)
        y = self.classifier.predict([x])[0]
        probs = list(self.classifier.predict_proba([x])[0])

        if self.ignore_t:
            probs = self.calc_with_t_prob(probs)
            if probs[y] < threshold:
                y = DbdReader.LABEL_TO_INDEX["T"]

        return y, probs
    
    @classmethod
    def calc_with_t_prob(cls, probs):
        prob_o = probs[DbdReader.LABEL_TO_INDEX["O"]]
        prob_x = probs[DbdReader.LABEL_TO_INDEX["X"]]
        prob_t = prob_x * -0.16721 + 0.36399  # heuristic, calculate alpha and intercept from breakdown data
        if prob_t > prob_x:
            remain = 1 - prob_t
            prob_x = remain * (prob_x / (prob_o + prob_x))
            prob_o = remain - prob_x
        else:
            prob_x = prob_x - prob_t

        _probs = probs + [prob_t]
        _probs[DbdReader.LABEL_TO_INDEX["X"]] = prob_x
        _probs[DbdReader.LABEL_TO_INDEX["O"]] = prob_o
        if (prob_t + prob_x + prob_o) - 1.0 > 1.0e-10:
            raise Exception("sum of probabilities does not equal to 1 {0}".format(prob_t + prob_x + prob_o))
        return _probs

    def extract_feature(self, session, utterance_pair):
        u, s = utterance_pair
        output, decoder_state, encoder_state = self.model_interface.predict(session, u, s)
        # output: selected bucket size x batch_size(=1 for predict) x model_size
        # state: selected bucket size x model_size

        summed_output = np.sum([np.mean(b, axis=0) for b in output], axis=0)  # take mean between batch
        last_encoder_state = encoder_state[-1]  # last state is most important
        last_decoder_state = decoder_state[-1]
        mul_state = np.multiply(last_encoder_state, last_decoder_state)

        features = np.concatenate((last_encoder_state, last_decoder_state))
        #features = np.concatenate((summed_output, mul_state))
        return features
