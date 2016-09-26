import random
import numpy as np
from model.dataset.loader import Loader


class Seq2SeqDataProcessor():

    def __init__(self, buckets):
        self.buckets = buckets
    
    def format(self, utterance_pair):
        source = list(utterance_pair[0])
        target = list(utterance_pair[1])

        bucket_id = self._detect_bucket(source, target + [0])  # + [0] is space for EOS
        s_adjusted, t_adjusted = self._adjust_to_bucket(source, target, self.buckets[bucket_id])
        return bucket_id, s_adjusted, t_adjusted
    
    def _detect_bucket(self, source, target):
        _id = -1
        for bucket_id, (source_size, target_size) in enumerate(self.buckets):
            if len(source) < source_size and len(target) < target_size:
                _id = bucket_id
                break
        return _id
    
    @classmethod
    def _adjust_to_bucket(cls, source, target, bucket):
        source_size = bucket[0]
        target_size = bucket[1]

        if len(source) <= source_size:
            source_pad = [Loader.PAD_ID] * (source_size - len(source))
            adjusted_source = list(reversed(source + source_pad))
        else:
            adjusted_source = list(reversed(source))[:source_size]  # trim after reverse (last word maybe more important)
        
        if len(target) <= target_size - 2:  # for GO & EOS TAG
            target_pad_size = target_size - len(target) - 2
            adjusted_target = [Loader.GO_ID] + target + [Loader.EOS_ID] + [Loader.PAD_ID] * target_pad_size
        else:
            adjusted_target = [Loader.GO_ID] + target[:target_size - 2] + [Loader.EOS_ID]
        
        return adjusted_source, adjusted_target

    def transpose_array(self, x, vertical_size, dtype=np.int32):
        """
        x = list.
        return [size x len(x)] numpy array
        """
        result = []
        for i in range(vertical_size):
            result.append(np.array([x[b][i] for b in range(len(x))], dtype=dtype))
        return result

    def batch_iter(self, training_data, batch_size, already_formatted=False):
        _formatted = training_data if already_formatted else [self.format(t) for t in training_data]

        bucket_boxs = [[] for _ in self.buckets]
        for b_id, source, target in _formatted:
            bucket_boxs[b_id].append([source, target])

        bucket_sizes = [len(bucket_boxs[b]) for b in range(len(self.buckets))]
        total_size = float(sum(bucket_sizes))
        buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

        while True:
            _random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > _random_number_01])
            encoder_size, decoder_size = self.buckets[bucket_id]

            encoder_inputs = []
            decoder_inputs = []
            weights = []
            for _ in range(batch_size):
                e, d = random.choice(bucket_boxs[bucket_id])
                if len(e) != encoder_size or len(d) != decoder_size:
                    raise Exception("formatted size does not match with bucket size.")
                encoder_inputs.append(e)
                decoder_inputs.append(d)
                weight = [1.0] * decoder_size
                for i in range(len(d)):
                    if i < len(d) - 1:
                        teacher = d[i + 1]
                    if i == len(d) - 1 or teacher == Loader.PAD_ID:
                        weight[i] = 0.0  # decode to last or PAD id is meaning less, so weight is 0
                weights.append(weight)
            
            batch_en = self.transpose_array(encoder_inputs, encoder_size)
            batch_de = self.transpose_array(decoder_inputs, decoder_size)
            batch_w = self.transpose_array(weights, decoder_size, dtype=np.float32)
            yield bucket_id, batch_en, batch_de, batch_w
