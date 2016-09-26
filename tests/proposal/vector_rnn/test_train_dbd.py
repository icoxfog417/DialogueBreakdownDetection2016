import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from model.proposal.vector_rnn.model import VectoredRNN, VectoredRNNInterface
from model.proposal.vector_rnn.trainer import VectorRNNTrainer
from model.proposal.conscious_rnn.detector import ConsciousDetector
from model.dataset.dbd_reader import DbdReader


class TestVectorRNNByDbdData(unittest.TestCase):
    DATA_DIR = ""
    TARGET_PATH = ""
    VECTOR_DATA_PATH = ""
    TRAIN_DIR = ""
    Reader = None
    vector_size = 200
    vocab_size = 2000
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    
    @classmethod
    def setUpClass(cls):
        # development data for last year
        cls.DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/dialog_log/2015/evaluation")
        cls.TARGET_PATH = os.path.join(os.path.dirname(__file__), "../../data/test_dbd_vector_rnn.txt")
        cls.VECTOR_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../run/vector_rnn/store/model_tokened.vec")
        cls.TRAIN_DIR = os.path.dirname(cls.TARGET_PATH) + "/training_vector_rnn"

        if not os.path.exists(cls.TRAIN_DIR):
            print("make training dir at {0}.".format(cls.TRAIN_DIR))
            os.makedirs(cls.TRAIN_DIR)
        
        cls.Reader = DbdReader(cls.DATA_DIR, cls.TARGET_PATH, max_vocabulary_size=cls.vocab_size, clear_when_exit=False)
        cls.Reader.init(cls.VECTOR_DATA_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.Reader.remove_files()
        cls.Reader = None
    
    def make_model(self, user_vocab, system_vocab, size):
        num_layers = 1

        model = VectoredRNN(
            source_vocab_size=len(user_vocab.vocab),
            target_vocab_size=len(system_vocab.vocab),
            size=size,
            num_layers=num_layers,
            name="vector_rnn"
        )

        return model

    def test_save_and_load_training(self):
        batch_size = 8

        dataset, user_vocab, system_vocab = self.Reader.get_dataset()
        vocab_vectors = self.Reader.user_loader.load_vocab_vectors()
        labels = self.Reader.get_labels()
        # train with save
        print("## Execute training with save.")
        with tf.Graph().as_default() as train:
            model = self.make_model(user_vocab, system_vocab, len(vocab_vectors[0]))
            trainer = VectorRNNTrainer(model, self.buckets, batch_size, vocab_vectors, self.TRAIN_DIR)
            with tf.Session() as sess:
                trainer.set_optimizer(sess)
                for x in trainer.train(sess, dataset, labels, check_interval=10, max_iteration=100):
                    pass

        print("## Now, load from saved model")
        with tf.Graph().as_default() as prediction:
            decode_user = lambda s: " ".join([tf.compat.as_str(user_vocab.rev_vocab[i]) for i in s])
            model = self.make_model(user_vocab, system_vocab, len(vocab_vectors[0]))
            samples = np.random.randint(len(dataset), size=5)
            with tf.Session() as sess:
                model_if = VectoredRNNInterface(model, self.buckets, vocab_vectors, model_path=self.TRAIN_DIR)
                model_if.build(sess, predict=True)
                for s in samples:
                    pair = dataset[s]
                    output, _, _ = model_if.predict(sess, pair[0], pair[1])
                    text = model_if.decode(output, system_vocab.rev_vocab)
                    print("{0} -> {1}".format(decode_user(pair[0]), text))


if __name__ == '__main__':
    unittest.main()
