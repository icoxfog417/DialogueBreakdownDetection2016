import numpy as np
from model.dataset.loader import Loader
from model.proposal.conscious_rnn.data_processor import ConsciousRNNDataProcessor


class VectoredRNNDataProcessor():

    def __init__(self, buckets, vocab_vectors, data_processor=None):
        self.buckets = buckets
        self.vocab_vectors = vocab_vectors
        if len(self.vocab_vectors) == 0:
            raise Exception("Vocabulary vector's size is 0.'")

        self._dp = data_processor if data_processor is not None else ConsciousRNNDataProcessor(buckets)

    def format(self, utterance_pair, dbd_data=None):
        if dbd_data is None:
            return self._dp.format(utterance_pair)
        else:
            return self._dp.format(utterance_pair, dbd_data)
    
    def get_vector_length(self):
        return len(self.vocab_vectors[0])

    def embed(self, word_id_sequence):
        """
        word_id_sequence is numpy array of np.int32 (word id sequence)
        """
        _word_id_sequence = word_id_sequence if isinstance(word_id_sequence, np.ndarray) else np.array(word_id_sequence)

        embeded = np.zeros((len(_word_id_sequence), self.get_vector_length()))
        for i, w_id in enumerate(_word_id_sequence):
            if w_id >= len(self.vocab_vectors):
                raise Exception("The word does not exist in vocabulary is detected ({0}).".format(u))
            embeded[i] = self.vocab_vectors[w_id]
        return embeded

    def batch_iter(self, training_data, batch_size, labels=()):
        if len(labels) > 0:
            for bucket_id, batch_en, batch_de, batch_w in self._dp.batch_iter(training_data, labels, batch_size):
                yield bucket_id, batch_en, batch_de, batch_w
        else:
            for bucket_id, batch_en, batch_de, batch_w in self._dp.batch_iter(training_data, batch_size):
                yield bucket_id, batch_en, batch_de, batch_w
