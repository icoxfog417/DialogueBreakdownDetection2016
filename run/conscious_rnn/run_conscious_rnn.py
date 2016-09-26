import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from model.baseline.seq2seq import Seq2Seq, Seq2SeqInterface
from model.baseline.probability_calculator import ProbabilityCalculator
from model.proposal.conscious_rnn.trainer import ConsciousRNNTrainer
from model.proposal.conscious_rnn.detector import ConsciousDetector
from model.dataset.dbd_reader import DbdReader
from model.baseline.analyzer import Analyzer


DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/dialog_log/all/train")
TEST_DIR = os.path.join(os.path.dirname(__file__), "../../data/dialog_log/all/test")
GET_PATH = lambda x: os.path.join(os.path.dirname(__file__), "./store/" + x)

TARGET_PATH = GET_PATH("conscious_target_data.txt")
TRAIN_PATH = GET_PATH("conscious_target_train.txt")
TEST_PATH = GET_PATH("conscious_target_test.txt")
TRAIN_DIR = GET_PATH("training")
SUBMIT_DIR = GET_PATH("submit")


# todo: have to read from setting file
_vocab_size_ = 2160
_buckets_ = [(5, 10), (10, 15), (20, 25), (40, 50)]
_size_ = 256
_num_layers_ = 1
_batch_size_ = 10
_learning_rate_ = 0.1
_check_interval_ = 1000
_max_iteration_ = -1


def make_model(user_vocab, system_vocab):
    seq2seq = Seq2Seq(
        source_vocab_size=len(user_vocab.vocab),
        target_vocab_size=len(system_vocab.vocab),
        size=_size_,
        num_layers=_num_layers_,
        name="conscious_rnn"
    )
    return seq2seq


def get_dataset_reader(train=True, full=True):
    reader = None
    if train:
        if full:
            reader = DbdReader(DATA_DIR, TARGET_PATH, max_vocabulary_size=_vocab_size_, clear_when_exit=False)
        else:
            reader = DbdReader(DATA_DIR, TRAIN_PATH, target_for_vocabulary=TARGET_PATH, max_vocabulary_size=_vocab_size_, filter="140", threshold=0.4)
    else:
        reader = DbdReader(TEST_DIR, TEST_PATH, target_for_vocabulary=TARGET_PATH, max_vocabulary_size=_vocab_size_, clear_when_exit=False)
    reader.init()
    return reader


def train():
    if not os.path.exists(TRAIN_DIR):
        print("make training dir at {0}.".format(TRAIN_DIR))
        os.makedirs(TRAIN_DIR)

    reader = get_dataset_reader()
    dataset, user_vocab, system_vocab = reader.get_dataset()
    labels = reader.get_labels()

    print("begin training.")
    model = make_model(user_vocab, system_vocab)
    trainer = ConsciousRNNTrainer(model, _buckets_, _batch_size_, TRAIN_DIR)
    with tf.Session() as sess:
        trainer.set_optimizer(sess, learning_rate=_learning_rate_)
        for x in trainer.train(sess, dataset, labels, check_interval=_check_interval_, max_iteration=_max_iteration_):
            pass


def predict():
    reader = get_dataset_reader(train=False)
    analyzer = Analyzer(reader)

    print("begin prediction")
    model = make_model(analyzer.user_vocab, analyzer.system_vocab)
    model_if = model.create_interface(_buckets_, model_path=TRAIN_DIR)

    samples = np.random.randint(len(analyzer.dataset), size=5)

    with tf.Session() as sess:
        model_if.build(sess, predict=True)  # because check seq2seq accuracy
        for s in samples:
            pair = dataset[s]
            output, _, _ = model_if.predict(sess, pair[0], pair[1])
            text = model_if.decode(output, analyzer.system_vocab.rev_vocab)
            print("{0} -> {1}".format(analyzer.get_user_utterance(pair[0]), text))


def train_detector(neural_net=False, ignore_t=False):
    reader = get_dataset_reader(full=False)
    analyzer = Analyzer(reader)

    model = make_model(analyzer.user_vocab, analyzer.system_vocab)
    model_if = model.create_interface(_buckets_, TRAIN_DIR)
        
    train_x, test_x, train_t, test_t = train_test_split(analyzer.dataset, analyzer.labels, test_size=0.2, random_state=42)

    y = []
    probs = []
    with tf.Session() as sess:
        detector = ConsciousDetector(sess, model_if, model_dir=TRAIN_DIR, ignore_t=ignore_t, load_if_exist=False)
        if not neural_net:
            detector.train(sess, train_x, train_t)
            y_p = [detector.predict(sess, p) for p in test_x]
            detector.save()
        else:
            prob_calculator = ProbabilityCalculator(sess, detector, layer_count=3)
            prob_calculator.train(sess, train_x, train_t, batch_size=200, epoch=20000, load_if_exist=False)
            y_p = [prob_calculator.predict(sess, p) for p in test_x]
            prob_calculator.save(sess)

    y = [yp[0] for yp in y_p]
    probs = [yp[1] for yp in y_p]
    analyzer.show_report(test_t, y, probs)
    analyzer.dump_result(os.path.dirname(TARGET_PATH), test_x, test_t, y, probs, "conscious_rnn_train")

def predict_detector(neural_net=False, ignore_t=False, submit=False):
    reader = get_dataset_reader(train=False)
    analyzer = Analyzer(reader)
    
    model = make_model(analyzer.user_vocab, analyzer.system_vocab)
    model_if = model.create_interface(_buckets_, TRAIN_DIR)

    with tf.Session() as sess:
        detector = ConsciousDetector(sess, model_if, model_dir=TRAIN_DIR, ignore_t=ignore_t, load_if_exist=True)
        if not neural_net:
            y_p = [detector.predict(sess, p) for p in analyzer.dataset]
        else:
            prob_calculator = ProbabilityCalculator(sess, detector)
            y_p = [prob_calculator.predict(sess, p) for p in analyzer.dataset]

    y = [yp[0] for yp in y_p]
    probs = [yp[1] for yp in y_p]

    samples = np.random.randint(len(analyzer.dataset), size=10)
    for i, d in enumerate([analyzer.dataset[s] for s in samples]):
        print("{0}({1}): {2} -> {3}".format(
            analyzer.label_names[y[i]], 
            analyzer.label_name_at(i), 
            analyzer.get_user_utterance(d[0]), 
            analyzer.get_system_utterance(d[1])))

    analyzer.show_report(analyzer.labels, y, probs)
    analyzer.dump_result(os.path.dirname(TARGET_PATH), analyzer.dataset, analyzer.labels, y, probs, "conscious_rnn_test")

    if submit:
        print("write submission data")
        analyzer.submit(SUBMIT_DIR, y, probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Seq2Seq model")
    parser.add_argument("--train", action="store_true", help="execute training")
    parser.add_argument("--predict", action="store_true", help="execute prediction")
    parser.add_argument("--detector", action="store_true", help="execute detector")
    parser.add_argument("--nn", action="store_true", help="neural net moddel", default=False)
    parser.add_argument("--ignore_t", action="store_true", help="ignore t label to train detector", default=False)
    parser.add_argument("--submit", action="store_true", help="make submission data", default=False)

    args = parser.parse_args()

    if args.train:
        if args.detector:
            train_detector(args.nn, args.ignore_t)
        else:
            train()
    elif args.predict:
        if args.detector:
            predict_detector(args.nn, args.ignore_t, args.submit)
        else:
            predict()
