import os
import json
from model.dataset.tokenizer import JapaneseTokenizer
from model.dataset.loader import Loader
from model.dataset.dbd_loader import DbdLoader

class CSVWriter(DbdLoader):

    def create_data(self, data_filter=None, threshold=0.7):
        if os.path.isfile(self.target_path):
            print("data file is already exist.")
            return 0

        files = [os.path.join(self.data_folder, c) for c in os.listdir(self.data_folder)]
        files = [f for f in files if os.path.isfile(f)]

        dialog = []
        for f in files:
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
                    dialog.append([user_uttr, bot_uttr] + a)
                    user_uttr = ""
                    bot_uttr = ""

        # write each utterances
        with open(self.user_path, "wb") as uf:
            with open(self.bot_path, "wb") as sf:
                for d in dialog:
                    write = lambda f, t: f.write((t + "\n").encode("utf-8"))
                    print(','.join([str(el) for el in d]))
#                    write(uf, d[0])
#                    write(sf, d[1])
