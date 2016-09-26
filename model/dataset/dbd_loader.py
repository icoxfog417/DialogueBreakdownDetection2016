import os
import json
from model.dataset.tokenizer import JapaneseTokenizer
from model.dataset.loader import Loader


class DbdLoader():
    FILE_NAME_FORMAT = "dbd_data_{0}.txt"

    def __init__(self, data_folder, target_path, target_for_vocabulary=""):
        self.data_folder = data_folder
        self.target_path = target_path
        self.user_path = Loader.add_suffix(self.target_path, "user")
        self.bot_path = Loader.add_suffix(self.target_path, "bot")
        self.target_for_vocabulary = []
        if target_for_vocabulary:
            # make link to data file that already exist
            self.target_for_vocabulary = [
                Loader.add_suffix(target_for_vocabulary, "user"),
                Loader.add_suffix(target_for_vocabulary, "bot")
            ]

    def remove_files(self):
        for f in [self.target_path, self.user_path, self.bot_path]:
            if os.path.isfile(f):
                os.remove(f)

    def create_data(self, filter="", threshold=-1):
        if os.path.isfile(self.target_path):
            print("data file is already exist.")
            return 0

        files = [os.path.join(self.data_folder, c) for c in os.listdir(self.data_folder)]
        files = [f for f in files if os.path.isfile(f)]

        dialog = []
        for f in files:
            f_id = os.path.basename(f).split(".")[0]

            if filter and f_id.startswith(filter):
                continue

            with open(f, "r", encoding="utf-8") as jf:
                j = json.load(jf)
                turns = j["turns"]
            
            user_uttr = ""
            bot_uttr = ""
            annotation = {}  # o, Δ, x, ungrammer
            for t in turns:
                if t["speaker"] == "U":
                    user_uttr = t["utterance"]
                elif t["speaker"] == "S":
                    bot_uttr = t["utterance"]
                    a = self._summarize_annotations(t["annotations"])
                    if threshold > 0 and max(a[1:4]) <= threshold:
                        continue
                    turn_index = t["turn-index"]
                    dialog.append([user_uttr, bot_uttr] + a + [turn_index, f_id])
                    user_uttr = ""
                    bot_uttr = ""

        # write summary file
        with open(self.target_path, "wb") as f:
            for d in dialog:
                line = "\t".join([str(x) for x in d]) + "\n"
                f.write(line.encode("utf-8"))

        # write each utterances
        with open(self.user_path, "wb") as uf:
            with open(self.bot_path, "wb") as sf:
                for d in dialog:
                    write = lambda f, t: f.write((t + "\n").encode("utf-8"))
                    write(uf, d[0])
                    write(sf, d[1])

    def create_loaders(self, max_vocabulary_size):
        if not self.user_path or not self.bot_path:
            raise Exception("You have to create data first.")
        
        tokenizer = JapaneseTokenizer()
        user_loader = Loader(self.user_path, tokenizer=tokenizer)
        if len(self.target_for_vocabulary) > 0:
            user_loader.vocabulary_path = Loader.make_vocab_path(self.target_for_vocabulary[0])

        bot_loader = Loader(self.bot_path, tokenizer=tokenizer)
        if len(self.target_for_vocabulary) > 0:
            bot_loader.vocabulary_path = Loader.make_vocab_path(self.target_for_vocabulary[1])
        
        loaders = [user_loader, bot_loader]
        for i, ld in enumerate(loaders):
            if i == 0:
                ld.create_vocabulary(max_vocabulary_size)
            else:
                ld.create_vocabulary(max_vocabulary_size)
                
            ld.create_token_ids()

        return loaders

    def _summarize_annotations(self, annotations):
        classes = {
            "O": 0,
            "T": 0,
            "X": 0,
        }
        un_grammer = 0

        counter = 0
        for a in annotations:
            label = a["breakdown"]
            if label in classes:
                classes[label] += 1
            un_grammer += 1 if a["ungrammatical-sentence"] == "O" else 0
            counter += 1
        
        conclusion = max(classes, key=classes.get)
        if counter != 0:
            for k in classes:
                classes[k] /= counter
            un_grammer /= counter
        
        summary = [
            conclusion,
            classes["O"],
            classes["T"],
            classes["X"],
            un_grammer
        ]

        return summary
