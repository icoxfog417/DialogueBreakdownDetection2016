import os
import numpy as np
import tensorflow as tf
from model.baseline.classifier import NeuralNetClassifier
from model.baseline.detector import Detector


class ProbabilityCalculator():

    def __init__(self, session, detector, layer_count=2, ignore_t=False):
        self.detector = detector
        self.ignore_t = ignore_t
        if detector.model_dir:
            self.model_dir = os.path.join(detector.model_dir, "./prob_model")
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        # confirm feature length
        dummy_features = self.detector.extract_feature(session, ([0], [0]))
        self.num_inputs = len(dummy_features)
        self.layers = [self.num_inputs] * layer_count
        self.classifier = NeuralNetClassifier(self.num_inputs, self.layers, num_class=2 if ignore_t else 3, model_dir=self.model_dir)
    
    def save(self, session):
        if not self.model_dir:
            raise Exception("Detector's model path is not specified.")
        else:
            self.classifier.save(session)
    
    def _make_model_path(self):
        return os.path.join(self.model_dir, "{0}.pkl".format(self.detector.name + "_calculator"))

    def train(self, session, utterance_pairs, labels, learning_rate=0.001, batch_size=10, epoch=100, load_if_exist=True, verbose=False):
        self.classifier.set_optimizer(session, learning_rate, load_if_exist=load_if_exist)
        probs = [lb.get_prob_vector(self.ignore_t) for lb in labels]
        X = []
        for pair in utterance_pairs:
            x = self.detector.extract_feature(session, pair)
            X.append(x)
        
        self.classifier.train(session, X, probs, batch_size=batch_size, epoch=epoch, verbose=verbose)
    
    def predict(self, session, utterance_pair, threshold=0.4, load_if_exist=True):
        self.classifier.build(session, load_if_exist)
        x = self.detector.extract_feature(session, utterance_pair)
        probs = self.classifier.predict(session, x)
        y = np.argmax(probs)

        if self.ignore_t:
            probs = Detector.calc_with_t_prob(probs)
            if max(probs) < threshold:
                y = DbdReader.LABEL_TO_INDEX["T"]

        return y, probs
