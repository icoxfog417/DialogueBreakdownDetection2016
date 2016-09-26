import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from model.baseline.seq2seq import Seq2Seq, Seq2SeqInterface
from model.proposal.conscious_rnn.trainer import ConsciousRNNTrainer
from model.proposal.conscious_rnn.detector import ConsciousDetector
from model.dataset.dbd_reader import DbdReader


class TestConsciousRNNByDbdData(unittest.TestCase):
    DATA_DIR = ""
    TARGET_PATH = ""
    TRAIN_DIR = ""
    Reader = None
    vocab_size = 1200
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    
    @classmethod
    def setUpClass(cls):
        # development data for last year
        cls.DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/dialog_log/2015/evaluation"))
        cls.TARGET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/test_dbd_cons_rnn.txt"))
        cls.TRAIN_DIR = os.path.abspath(os.path.dirname(cls.TARGET_PATH) + "/training_cons_rnn")

        if not os.path.exists(cls.TRAIN_DIR):
            print("make training dir at {0}.".format(cls.TRAIN_DIR))
            os.makedirs(cls.TRAIN_DIR)
        
        cls.Reader = DbdReader(cls.DATA_DIR, cls.TARGET_PATH, max_vocabulary_size=cls.vocab_size, clear_when_exit=False)
        cls.Reader.init()

    @classmethod
    def tearDownClass(cls):
        cls.Reader.remove_files()
        cls.Reader = None
    
    def make_model(self, user_vocab, system_vocab):
        size = 256
        num_layers = 1

        seq2seq = Seq2Seq(
            source_vocab_size=len(user_vocab.vocab),
            target_vocab_size=len(system_vocab.vocab),
            size=size,
            num_layers=num_layers,
            name="conscious_rnn"
        )

        return seq2seq

    def xtest_save_and_load_training(self):
        batch_size = 8

        dataset, user_vocab, system_vocab = self.Reader.get_dataset()
        labels = self.Reader.get_labels()
        # train with save
        print("## Execute training with save.")
        with tf.Graph().as_default() as train:
            model = self.make_model(user_vocab, system_vocab)
            trainer = ConsciousRNNTrainer(model, self.buckets, batch_size, self.TRAIN_DIR)
            with tf.Session() as sess:
                trainer.set_optimizer(sess)
                for x in trainer.train(sess, dataset, labels, check_interval=10, max_iteration=100):
                    pass

        print("## Now, load from saved model")
        with tf.Graph().as_default() as prediction:
            decode_user = lambda s: " ".join([tf.compat.as_str(user_vocab.rev_vocab[i]) for i in s])
            model = self.make_model(user_vocab, system_vocab)
            samples = np.random.randint(len(dataset), size=5)
            with tf.Session() as sess:
                model_if = Seq2SeqInterface(model, self.buckets, model_path=self.TRAIN_DIR)
                model_if.build(sess, predict=True)
                for s in samples:
                    pair = dataset[s]
                    output, _, _ = model_if.predict(sess, pair[0], pair[1])
                    text = model_if.decode(output, system_vocab.rev_vocab)
                    print("{0} -> {1}".format(decode_user(pair[0]), text))

    def test_detector(self):
        dataset, user_vocab, system_vocab = self.Reader.get_dataset()
        decode_user = lambda s: " ".join([tf.compat.as_str(user_vocab.rev_vocab[i]) for i in s])

        _labels = self.Reader.get_labels()
        labels = [lb.label for lb in _labels]
        model = self.make_model(user_vocab, system_vocab)
        model_if = model.create_interface(self.buckets, self.TRAIN_DIR)
        
        train_x, test_x, train_t, test_t = train_test_split(dataset, labels, test_size=0.2, random_state=42)

        judges = []
        label_names = DbdReader.get_label_names()
        with tf.Session() as sess:
            detector = ConsciousDetector(sess, model_if)
            detector.train(sess, train_x, train_t)
            y = [detector.predict(sess, p) for p in test_x]

            for i, p in enumerate(test_x):
                output, _, _ = detector.model_interface.predict(sess, p[0], p[1])
                judges.append([
                    label_names[y[i]], decode_user(p[0]), detector.model_interface.decode(output, system_vocab.rev_vocab)
                ])
        
        report = classification_report(test_t, y, target_names=label_names)
        print(report)

        for i, j in enumerate(judges):
            print("{0}({1}): {2} -> {3}".format(j[0], label_names[test_t[i]], j[1], j[2]))


if __name__ == '__main__':
    unittest.main()
