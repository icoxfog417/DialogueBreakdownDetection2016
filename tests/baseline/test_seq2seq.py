import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from model.baseline.seq2seq import Seq2Seq, Seq2SeqInterface


class TestSeq2Seq(unittest.TestCase):

    def test_forward(self):
        source_vocab_size = 10
        target_vocab_size = 15
        size = 12
        buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

        seq2seq = Seq2Seq(source_vocab_size, target_vocab_size, size)
        interface = Seq2SeqInterface(seq2seq, buckets)

        with tf.Session() as sess:
            interface.build(sess)
            output, d_state, e_state = interface.predict(sess, [1, 2, 3, 4], [4, 5, 6, 7, 8, 9])
            rev_dict = [str(i) for i in range(target_vocab_size)]
            text = interface.decode(output, rev_dict)
            print(text)


if __name__ == '__main__':
    unittest.main()
