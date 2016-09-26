from collections import namedtuple
from model.dataset.dbd_loader import DbdLoader
from model.dataset.reader import ParallelReader


Vocabulary = namedtuple("Vocabulary", ["vocab", "rev_vocab"])

class DbdData():

    def __init__(self, user_utterance, system_utterance, label, o_prob, t_prob, x_prob, un_grammer_prob, turn_index=0, dialog_id=""):
        self.user_utterance = user_utterance
        self.system_utterance = system_utterance
        if label not in DbdReader.LABEL_TO_INDEX:
            raise Exception("unknown label {0} is found in label file.".format(label))
        self.label = DbdReader.LABEL_TO_INDEX[label]
        self.o_prob = float(o_prob)
        self.t_prob = float(t_prob)
        self.x_prob = float(x_prob)
        self.un_grammer_prob = float(un_grammer_prob)
        self.turn_index = turn_index
        self.dialog_id = dialog_id
    
    def get_label(self, ignore_t=False):
        if ignore_t and self.get_label_text() == "T":
            return DbdReader.LABEL_TO_INDEX["X"]
        else:
            return self.label

    def get_one_hot_vector(self, ignore_t=False):
        if not ignore_t:
            return [1 if i == self.label else 0 for i in range(len(DbdReader.LABEL_TO_INDEX))]
        else:
            return [1 if i == self.label else 0 for i in range(2)]

    def get_prob_vector(self, ignore_t=False):
        _dic = DbdReader.LABEL_TO_INDEX
        v = [0.0] * len(_dic)
        for k in _dic:
            if k == "O":
                v[_dic[k]] = self.o_prob
            elif k == "X":
                v[_dic[k]] = self.x_prob
            elif k == "T":
                v[_dic[k]] = self.t_prob
        
        if not ignore_t:
            return v
        else:
            v[_dic["X"]] += v[_dic["T"]]
            return v[:2]

    def get_label_text(self):
        text = ""
        for lb in DbdReader.LABEL_TO_INDEX:
            if DbdReader.LABEL_TO_INDEX[lb] == self.label:
                text = lb
        return text
    
    def __str__(self):
        format = "{0}: {1} -> {2} ({3}, {4}, {5}, grammer={6})"
        return format.format(self.get_label_text(), self.user_utterance, self.system_utterance, self.o_prob, self.t_prob, self.x_prob, self.un_grammer_prob)


class DbdReader():
    LABEL_TO_INDEX = {"O": 1, "X": 0, "T": 2}

    def __init__(self, dbd_data_folder, target_path, target_for_vocabulary="", max_vocabulary_size=1000, filter="", threshold=-1, clear_when_exit=True):
        self.loader = DbdLoader(dbd_data_folder, target_path, target_for_vocabulary)
        self.max_vocabulary_size = max_vocabulary_size
        self.filter = filter
        self.threshold = threshold
        self.clear_when_exit = clear_when_exit
        self.user_loader = None
        self.bot_loader = None
    
    def remove_files(self):
        self.loader.remove_files()
        if self.user_loader:
            self.user_loader.remove_files()
            self.user_loader = None
        if self.bot_loader:
            self.bot_loader.remove_files()
            self.bot_loader = None
    
    @classmethod
    def get_label_names(cls):
        names = [""] * len(cls.LABEL_TO_INDEX)
        for k in cls.LABEL_TO_INDEX:
            names[cls.LABEL_TO_INDEX[k]] = k
        return names

    @classmethod
    def make_prob_dict(cls, prob_array):
        result = {}
        for k in cls.LABEL_TO_INDEX:
            result[k] = prob_array[cls.LABEL_TO_INDEX[k]]
        return result

    def __enter__(self):
        if not self.user_loader or not self.bot_loader:
            return self.init()
        else:
            return self
    
    def init(self, vector_data_path="", vector_data_vocab_path=""):
        self.loader.create_data(self.filter, self.threshold)  # if already exist, the file is not created
        self.user_loader, self.bot_loader = self.loader.create_loaders(
            self.max_vocabulary_size)
        if vector_data_path:
            self.user_loader.vector_data_path = vector_data_path
            self.user_loader.create_vocab_vectors(vector_data_vocab_path=vector_data_vocab_path)
        return self

    def get_dataset(self):
        user_vocab, user_rev_vocab = self.user_loader.load_vocabulary()
        bot_vocab, bot_rev_vocab = self.bot_loader.load_vocabulary()

        reader = ParallelReader(self.user_loader.ids_path, self.bot_loader.ids_path)
        dataset = reader.read_to_list()

        return dataset, Vocabulary(user_vocab, user_rev_vocab), Vocabulary(bot_vocab, bot_rev_vocab)
    
    def get_labels(self):
        labels = []
        with open(self.loader.target_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split("\t")
                if line.startswith("\t"):
                    tokens = [""] + tokens
                label = DbdData(*tokens)  # caution! if file format changes, you have to change here
                labels.append(label)
        return labels

    def __exit__(self, type, value, traceback):
        if self.clear_when_exit:
            self.loader.remove_files()
            self.user_loader.remove_files()
            self.bot_loader.remove_files()
            self.user_loader = None
            self.bot_loader = None
