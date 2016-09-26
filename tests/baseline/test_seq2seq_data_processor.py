import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from model.dataset.loader import Loader
from model.baseline.seq2seq_data_processor import Seq2SeqDataProcessor


class TestSeq2SeqDataProcessor(unittest.TestCase):

    def test_select_bucket(self):
        bucket_1 = (2, 3)
        bucket_2 = (5, 7)
        dp = Seq2SeqDataProcessor([bucket_1, bucket_2])

        bucket_id = dp._detect_bucket([1], [1, 2])
        self.assertEqual(0, bucket_id)
        bucket_id = dp._detect_bucket([1, 2], [1, 2, 3, 4])
        self.assertEqual(1, bucket_id)
        bucket_id = dp._detect_bucket([1, 2, 3], [1, 2, 3])
        self.assertEqual(1, bucket_id)
        bucket_id = dp._detect_bucket([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(1, bucket_id)

    def test_padding(self):
        bucket = (5, 7)
        dp = Seq2SeqDataProcessor([bucket])

        adjusted_s, adjusted_t = dp._adjust_to_bucket([1, 2], [3, 4, 5], bucket)
        self.assertEqual(bucket[0], len(adjusted_s))
        self.assertEqual(bucket[1], len(adjusted_t))
        no_pad = [t for t in adjusted_t if t is not Loader.PAD_ID]
        self.assertEqual(Loader.GO_ID, no_pad[0])
        self.assertEqual(Loader.EOS_ID, no_pad[len(no_pad) -1])

        adjusted_s, adjusted_t = dp._adjust_to_bucket(list(range(6)), list(range(8)), bucket)
        self.assertEqual(bucket[0], len(adjusted_s))
        self.assertEqual(bucket[1], len(adjusted_t))
        no_pad = [t for t in adjusted_t if t is not Loader.PAD_ID]
        self.assertEqual(Loader.GO_ID, no_pad[0])
        self.assertEqual(Loader.EOS_ID, no_pad[len(no_pad) -1])


if __name__ == '__main__':
    unittest.main()
