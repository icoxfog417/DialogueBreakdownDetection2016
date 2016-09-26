import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from model.baseline.seq2seq import Seq2Seq, Seq2SeqInterface
from model.baseline.seq2seq_trainer import Seq2SeqTrainer
from model.baseline.detector import Detector
from model.dataset.dbd_reader import DbdReader


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dialog_log/all/train"))
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dialog_log/all/test"))
TARGET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./store/seq2seq_target_data.txt"))
TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./store/seq2seq_train_detector.txt"))
TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./store/seq2seq_target_test.txt"))
TRAIN_DIR = os.path.abspath(os.path.dirname(TARGET_PATH) + "/training")


# todo: have to read from setting file
_vocab_size_ = 1000
_buckets_ = [(5, 10), (10, 15), (20, 25), (40, 50)]
_size_ = 128
_num_layers_ = 1
_batch_size_ = 8
_check_interval_ = 1000
_max_iteration_ = 100000


def make_model(user_vocab, system_vocab):
    seq2seq = Seq2Seq(
        source_vocab_size=len(user_vocab.vocab),
        target_vocab_size=len(system_vocab.vocab),
        size=_size_,
        num_layers=_num_layers_
    )
    return seq2seq


def train():
    if not os.path.exists(TRAIN_DIR):
        print("make training dir at {0}.".format(TRAIN_DIR))
        os.makedirs(TRAIN_DIR)

    reader = DbdReader(DATA_DIR, TARGET_PATH, max_vocabulary_size=_vocab_size_, threshold=0.6, clear_when_exit=False)
    reader.init()
    dataset, user_vocab, system_vocab = reader.get_dataset()

    print("begin training.")
    model = make_model(user_vocab, system_vocab)
    trainer = Seq2SeqTrainer(model, _buckets_, _batch_size_, TRAIN_DIR)
    with tf.Session() as sess:
        trainer.set_optimizer(sess)
        for x in trainer.train(sess, dataset, check_interval=_check_interval_, max_iteration=_max_iteration_):
            pass


def predict():
    reader = DbdReader(TEST_DIR, TEST_PATH, target_for_vocabulary=TARGET_PATH, max_vocabulary_size=_vocab_size_, clear_when_exit=False)
    reader.init()
    dataset, user_vocab, system_vocab = reader.get_dataset()

    print("begin prediction")
    decode_user = lambda s: " ".join([tf.compat.as_str(user_vocab.rev_vocab[i]) for i in s])
    model = make_model(user_vocab, system_vocab)
    model_if = model.create_interface(_buckets_, model_path=TRAIN_DIR)

    samples = np.random.randint(len(dataset), size=5)

    with tf.Session() as sess:
        model_if.build(sess, predict=True)
        for s in samples:
            pair = dataset[s]
            output, _, _ = model_if.predict(sess, pair[0], pair[1])
            text = model_if.decode(output, system_vocab.rev_vocab)
            print("{0} -> {1}".format(decode_user(pair[0]), text))


def classify():
    reader = DbdReader(DATA_DIR, TRAIN_PATH, target_for_vocabulary=TARGET_PATH, max_vocabulary_size=_vocab_size_, filter="140", threshold=0.6, clear_when_exit=False)
    reader.init()
    dataset, user_vocab, system_vocab = reader.get_dataset()

    labels = reader.get_labels()
    model = make_model(user_vocab, system_vocab)
    model_if = model.create_interface(_buckets_, TRAIN_DIR)
        
    train_x, test_x, train_t, test_t = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    with tf.Session() as sess:
        detector = Detector(sess, model_if)
        detector.train(sess, train_x, train_t)
        y = [detector.predict(sess, p) for p in test_x]
        y = [lb for lb, prob in y]
    
    report = classification_report([lb.label for lb in test_t], y, target_names=DbdReader.get_label_names())
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Seq2Seq model")
    parser.add_argument("--train", action="store_true", help="execute training")
    parser.add_argument("--predict", action="store_true", help="execute prediction")
    parser.add_argument("--classify", action="store_true", help="execute classification")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict()
    elif args.classify:
        classify()
