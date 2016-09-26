import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from model.baseline.seq2seq import Seq2Seq
from model.baseline.seq2seq_trainer import Seq2SeqTrainer


class TestSeq2SeqTrainer(unittest.TestCase):

    def test_training(self):
        source_vocab_size = 10
        target_vocab_size = 15
        size = 12
        batch_size = 4
        buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

        seq2seq = Seq2Seq(source_vocab_size, target_vocab_size, size)
        trainer = Seq2SeqTrainer(seq2seq, buckets, batch_size)

        test_data = self.make_test_data()

        with tf.Session() as sess:
            trainer.set_optimizer(sess)
            for x in trainer.train(sess, test_data, max_iteration=5):
                print(x)
    
    def make_test_data(self):
        ask_and_reply = [
            [[1, 2, 3, 4], [3, 4, 6, 7, 8]],
            [[0, 9, 1, 5, 8, 7, 3], [1, 3, 5, 5, 8]],
            [[0, 9, 1, 5, 8, 7, 1, 3, 3, 1, 5, 6], [0, 9, 1, 5, 8, 10, 9, 3, 3, 1, 5, 6]]
        ]
        return ask_and_reply


if __name__ == '__main__':
    unittest.main()
