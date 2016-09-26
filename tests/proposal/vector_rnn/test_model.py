import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import tensorflow as tf
from model.proposal.vector_rnn.model import VectoredRNN, VectoredRNNInterface


class TestModel(unittest.TestCase):

    def test_forward(self):
        vocab_size = 10
        vector_size = 15

        rev_vocabs = ["w-{0}".format(i) for i in range(vocab_size)]
        vocab_vector = [[1 if e == v else 0 for e in range(vector_size)] for v in range(vocab_size)]
        buckets = [(5, 10), (10, 20)]

        model = VectoredRNN(vocab_size, vocab_size, size=vector_size)
        model_if = VectoredRNNInterface(model, buckets, vocab_vector)

        with tf.Session() as s:
            model_if.build(s, predict=False)
            utterance_pair = [(1, 2, 3), (6, 7, 9, 3)]
            output, _, _ = model_if.predict(s, utterance_pair[0], utterance_pair[1])
            text = model_if.decode(output, rev_vocabs)
            self.assertTrue(text)
            print(text)


if __name__ == "__main__":
    unittest.main()
