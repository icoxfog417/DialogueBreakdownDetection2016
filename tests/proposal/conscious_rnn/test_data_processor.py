import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import numpy as np
import tensorflow as tf
from model.proposal.conscious_rnn.data_processor import ConsciousRNNDataProcessor
from model.dataset.dbd_reader import DbdData
from model.dataset.loader import Loader


class TestConsciousRNNDataProcessor(unittest.TestCase):

    def test_format(self):
        bucket_1 = (2, 4)
        bucket_2 = (5, 7)
        user_utterance = [1, 2, 3]
        system_utterance = [4, 5, 6, 7, 8]
        dp = ConsciousRNNDataProcessor([bucket_1, bucket_2])

        b_id, f_user, f_system = dp.format((user_utterance, system_utterance))
        
        self.assertEqual(1, b_id)
        self.assertEqual(bucket_2[0], len(f_user))
        self.assertEqual(bucket_2[1], len(f_system))

    def test_format_with_label(self):
        bucket_1 = (5, 7)
        bucket_2 = (10, 14)
        user_utterance = [1, 2, 3]
        system_utterance = [4, 5, 6, 7, 8]
        label = DbdData("today is fine", "my name is taro suzuki", "O", 0.4, 0.5, 0.1, 0.8)

        dp = ConsciousRNNDataProcessor([bucket_1, bucket_2])
        b_id, f_user, f_system = dp.format((user_utterance, system_utterance), label)
        
        self.assertEqual(1, b_id)
        self.assertEqual(bucket_2[0], len(f_user))
        self.assertEqual(bucket_2[1], len(f_system))

        no_pad = lambda a: [x for x in a if x != Loader.PAD_ID]
        self.assertEqual(list(reversed(user_utterance)), no_pad(f_user))
        self.assertEqual(Loader.GO_ID, f_system[0])
        self.assertEqual(Loader.EOS_ID, no_pad(f_system)[-1])
        self.assertTrue(no_pad(f_system)[-2] in [Loader.OK_ID, Loader.NG_ID, Loader.NEITHER_ID])

        # confirm prob of label
        label_count = [0, 0, 0]
        try_count = 1000
        for i in range(try_count):
            b_id, f_user, f_system = dp.format((user_utterance, system_utterance), label)
            label_id = no_pad(f_system)[-2]
            _label = -1
            if label_id == Loader.OK_ID:
                _label = 0
            elif label_id == Loader.NEITHER_ID:
                _label = 1
            elif label_id == Loader.NG_ID:
                _label = 2
            label_count[_label] += 1
        
        print("{0} has to close to {1}.".format([c / try_count for c in label_count], [0.4, 0.5, 0.1]))


if __name__ == "__main__":
    unittest.main()
