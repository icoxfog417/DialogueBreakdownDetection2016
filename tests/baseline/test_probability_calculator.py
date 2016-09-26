import unittest
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from model.baseline.seq2seq import Seq2Seq, Seq2SeqInterface
from model.baseline.detector import Detector
from model.baseline.probability_calculator import ProbabilityCalculator


class TestProbabilityCalculator(unittest.TestCase):
    MODEL_DIR = ""

    @classmethod
    def setUpClass(cls):
        cls.MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_prob_calculator"))
        if not os.path.exists(cls.MODEL_DIR):
            print("create folder for detector model")
            os.makedirs(cls.MODEL_DIR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.MODEL_DIR)

    def test_detector(self):
        buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        model = Seq2Seq(
            source_vocab_size=10,
            target_vocab_size=10,
            size=5,
        )
        model_if = model.create_interface(buckets)

        with tf.Session() as sess:
            detector = Detector(sess, model_if, self.MODEL_DIR)
            calculator = ProbabilityCalculator(sess, detector, layer_count=2)

            calculator.train(sess,
                [
                [(1, 2, 3), (4, 5, 6)],
                [(7, 8, 9), (0, 2, 3)],
                [(1, 2, 3), (4, 5, 6)]
                ],
                [
                    [0.1, 0.3, 0.6], 
                    [0.8, 0.1, 0.1], 
                    [0.2, 0.5, 0.3]
                ]            
            )
            output = calculator.predict(sess, [(1, 2, 3), (7, 8, 9)])
            print(output)
            self.assertEqual(3, len(output))  # 3 is dialog breakdown types
            calculator.save(sess)


if __name__ == "__main__":
    unittest.main()
