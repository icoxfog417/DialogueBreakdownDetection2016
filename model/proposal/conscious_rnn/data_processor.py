import random
import numpy as np
from model.dataset.loader import Loader
from model.dataset.dbd_reader import DbdData
from model.baseline.seq2seq_data_processor import Seq2SeqDataProcessor


class ConsciousRNNDataProcessor():

    def __init__(self, buckets):
        self.buckets = buckets
        self._dp = Seq2SeqDataProcessor(buckets)
    
    def format(self, utterance_pair, dbd_data=None):
        source = list(utterance_pair[0])
        target = list(utterance_pair[1])

        dummy_target = target + [0] * (1 if dbd_data is None else 2)  # when dbd_data is exist, add breakdown tag
        bucket_id = self._dp._detect_bucket(source, dummy_target)
        s_adjusted, t_adjusted = self._adjust_to_bucket(source, target, dbd_data, self.buckets[bucket_id])
        return bucket_id, s_adjusted, t_adjusted
    
    @classmethod
    def _adjust_to_bucket(cls, source, target, dbd_data, bucket):
        if dbd_data is None:
            return Seq2SeqDataProcessor._adjust_to_bucket(source, target, bucket)

        source_size = bucket[0]
        target_size = bucket[1]
        label_dist = []
        accume = 0.0
        for v_id, p in zip(
            [Loader.OK_ID, Loader.NG_ID],
            [dbd_data.o_prob, dbd_data.x_prob + dbd_data.t_prob]):
            # deal t as x
            accume += p
            label_dist.append([v_id, accume])
        
        _rand = np.random.random_sample()
        _label_index = min([i for i in range(len(label_dist)) if _rand < label_dist[i][1]])
        breakdown_label = [label_dist[_label_index][0]]

        if len(source) <= source_size:
            source_pad = [Loader.PAD_ID] * (source_size - len(source))
            adjusted_source = list(reversed(source + source_pad))
        else:
            adjusted_source = list(reversed(source))[:source_size]  # trim after reverse (last word maybe more important)

        if len(target) <= target_size - 3:
            target_pad_size = target_size - len(target) - 3
            adjusted_target = [Loader.GO_ID] + target + breakdown_label + [Loader.EOS_ID] + [Loader.PAD_ID] * target_pad_size
        else:
            adjusted_target = [Loader.GO_ID] + target[:target_size - 3] + breakdown_label + [Loader.EOS_ID]

        return adjusted_source, adjusted_target

    def batch_iter(self, training_data, labels, batch_size):
        _formatted = [self.format(t, lb) for t, lb in zip(training_data, labels)]

        for bucket_id, batch_en, batch_de, batch_w in self._dp.batch_iter(_formatted, batch_size, already_formatted=True):
            yield bucket_id, batch_en, batch_de, batch_w
