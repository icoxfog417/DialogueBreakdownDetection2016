import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from model.dataset.loader import Loader
from model.dataset.dbd_reader import DbdReader


class Analyzer():

    def __init__(self, dbd_reader):
        self.dataset, self.user_vocab, self.system_vocab = dbd_reader.get_dataset()
        self.labels = dbd_reader.get_labels()
        self.label_names = dbd_reader.get_label_names()
    
    def label_name_at(self, index):
        return self.label_names[self.labels[index].label]

    def get_user_utterance(self, token_ids):
        return " ".join([tf.compat.as_str(self.user_vocab.rev_vocab[i]) for i in token_ids])

    def get_system_utterance(self, token_ids):
        return " ".join([tf.compat.as_str(self.system_vocab.rev_vocab[i]) for i in token_ids])
    
    def show_report(self, dbd_data, labels, probs=()):
        t_labels = [t.label for t in dbd_data]
        t_probs = [t.get_prob_vector() for t in dbd_data]

        score = accuracy_score(t_labels, labels)
        if len(probs) > 0:
            mse = mean_squared_error(t_probs, probs)

        report = classification_report(t_labels, labels, target_names=self.label_names)
        print("accuracy={0} mse={1} (when mse=-1, probability is not set.)".format(score, mse))
        print(report)

    def dump_result(self, file_dir, utterance_pairs, dbd_data, labels, probs=(), name=""):
        if not os.path.exists(file_dir):
            raise Exception("analyzed file directory does not exist.")
        f_path = os.path.join(file_dir, "analyzed.txt" if not name else name + ".txt")

        results = []
        count_ids = lambda arr, _id: sum([1 if e == _id else 0 for e in arr])
        for i, pair in enumerate(utterance_pairs):
            user = self.get_user_utterance(pair[0])
            system = self.get_system_utterance(pair[1])

            label_text = self.label_names[labels[i]]
            teacher_text = self.label_names[dbd_data[i].label]
            max_prob_match = "-"
            probs_array = []
            if len(probs) > 0:
                ps = probs[i]
                tps = dbd_data[i].get_prob_vector()
                if np.argmax(ps) == np.argmax(tps):
                    max_prob_match = "O"
                p_max = probs[i][np.argmax(ps)]
                t_max = tps[np.argmax(tps)]
                probs_array = [t_max] + [p_max] + list(tps) + list(ps)
            
            unknown_in_user = count_ids(pair[0], Loader.UNK_ID)
            unknown_in_system = count_ids(pair[1], Loader.UNK_ID)

            line = [
                "O" if teacher_text == label_text else "X",
                max_prob_match,
                teacher_text,
                label_text,
                user,
                system,
                unknown_in_user,
                unknown_in_system
            ]
            line += probs_array
            results.append(line)
        
        with open(f_path, "w", encoding="utf-8") as f:
            header = [
                "label match",
                "prob match",
                "teacher label",
                "predicted label",
                "user utterance",
                "system utterance",
                "unknown in user",
                "unknown in system",
            ]
            if len(probs) > 0:
                prob_header = sum([["{0} {1} prob".format(k, n) for n in self.label_names] for k in ("teacher", "predicted")], [])
                prob_header = ["teacher max prob", "predicted max prob"] + prob_header
                header += prob_header
                
            write = lambda _ln: f.write("\t".join([str(e) for e in _ln]) + "\n")
            write(header)
            for ln in results:
                write(ln)
    
    def submit(self, submission_dir, y, probs):
        if not os.path.exists(submission_dir):
            print("make directory for submission at {0}".format(submission_dir))
            os.mkdir(submission_dir)

        submission = {}
        for i, ld in enumerate(self.labels):
            if not ld.dialog_id in submission:
                submission[ld.dialog_id] = []
            
            label = self.label_names[y[i]]
            ps = DbdReader.make_prob_dict(probs[i])
            submission[ld.dialog_id].append({
                "turn-index": ld.turn_index,
                "labels": [{
                    "breakdown": label,
                    "prob-O": ps["O"], 
                    "prob-T": ps["T"], 
                    "prob-X": ps["X"]
                }]
            })
        
        for s in submission:
            path = os.path.join(submission_dir, "{0}.labels.json".format(s))
            with open(path, "w", encoding="utf-8") as f:
                content = {
                    "dialogue-id": s,
                    "turns": submission[s]
                }
                json_str = json.dumps(content, indent=4)
                f.write(json_str)
    