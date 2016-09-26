import unittest
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from model.baseline.classifier import NeuralNetClassifier


class TestClassifier(unittest.TestCase):
    MODEL_DIR = ""

    @classmethod
    def setUpClass(cls):
        cls.MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_classifier"))
        if not os.path.exists(cls.MODEL_DIR):
            print("create folder for detector model")
            os.makedirs(cls.MODEL_DIR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.MODEL_DIR)

    def test_nn_classifier(self):
        size = 15
        batch_size = 10
        data_length = 1000
        model = NeuralNetClassifier(size, [20, 20], model_dir=self.MODEL_DIR)
        inputs_array, labels_array = self.make_test_data(data_length, size)

        with tf.Session() as sess:
            model.set_optimizer(sess)
            model.train(sess, inputs_array, labels_array, batch_size, epoch=5, verbose=True)
            
            # predict
            samples = np.random.randint(len(inputs_array), size=5)
            for s in samples:
                o = model.predict(sess, inputs_array[s])
                print(o)

            model.save(sess)

    def make_test_data(self, data_length, size):
        word_dist_of_label = [
            (0.2, 0.1),
            (0.5, 0.1),
            (0.8, 0.1),
        ]

        inputs_array = []
        labels_array = []
        num_class = 3
        labels = np.random.randint(num_class, size=data_length)  # 3 is class number
        labels_array = [[1.0 if i == lb else 0.0 for i in range(num_class)] for lb in labels]
        for lb in labels:
            dist = word_dist_of_label[lb]
            samples = np.random.normal(dist[0], dist[1], size)
            inputs_array.append(samples)
        
        return inputs_array, labels_array


if __name__ == '__main__':
    unittest.main()
