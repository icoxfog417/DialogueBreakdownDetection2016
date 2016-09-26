import os
import re
from tensorflow.python.platform import gfile
from model.dataset.tokenizer import BasicTokenizer


class Loader():
    _PAD = b"_PAD"
    _GO = b"_GO"
    _EOS = b"_EOS"
    _UNK = b"_UNK"
    _OK = b"_O"
    _NG = b"_X"
    _NEITHER = b"_T"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK, _OK, _NG, _NEITHER]
    _DIGIT_RE = re.compile(br"\d")
    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    OK_ID = 4
    NG_ID = 5
    NEITHER_ID = 6
    VOCAB_SUFFIX = "vocab"
    IDS_SUFFIX = "ids"
    VECS_SUFFIX = "vecs"

    def __init__(self, data_path, vocabulary_path="", vector_data_path="", tokenizer=None, normalize_digits=True):
        self.data_path = data_path
        self.vector_data_path = vector_data_path
        self.vocabulary_path = vocabulary_path if vocabulary_path else self.make_vocab_path(data_path)
        self.ids_path = self.add_suffix(data_path, self.IDS_SUFFIX)
        self.vecs_path = self.add_suffix(data_path, self.VECS_SUFFIX)
        self.tokenizer = tokenizer if tokenizer is not None else BasicTokenizer()
        self.normalize_digits = normalize_digits
    
    @classmethod
    def make_vocab_path(cls, path):
        return cls.add_suffix(path, cls.VOCAB_SUFFIX)

    @classmethod
    def add_suffix(cls, file_path, suffix):
        path, ext = os.path.splitext(file_path)
        added = path + "_" + suffix + ext
        return added
    
    def remove_files(self):
        for f in [self.vocabulary_path, self.ids_path, self.vecs_path]:
            if os.path.isfile(f):
                os.remove(f)
    
    def tokenize(self, line):
        tokens = self.tokenizer.tokenize(line)
        if self.normalize_digits:
            num_replaced = [re.sub(self._DIGIT_RE, b"0", w) for w in tokens] # normalize all digits to zero
        return num_replaced

    def create_vocabulary(self, max_vocabulary_size, progress_interval=100000):
        if gfile.Exists(self.vocabulary_path):
            print("vocabulary file already exist")
            return 0
        else:
            print("Creating vocabulary to %s from data %s" % (self.vocabulary_path, self.data_path))

        vocab = {}
        with gfile.GFile(self.data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % progress_interval == 0:
                    print("  processing line %d" % counter)
                tokens = self.tokenize(line)
                for w in tokens:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1

        vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(self.vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

    def load_vocabulary(self, vocabulary_path=""):
        _vp = vocabulary_path if vocabulary_path else self.vocabulary_path
        return self.read_vocabulary(_vp)
    
    @classmethod
    def read_vocabulary(self, vocabulary_path):
        if gfile.Exists(vocabulary_path):
            rev_vocab = []
            with gfile.GFile(vocabulary_path, mode="rb") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", _vp)        

    def sentence_to_token_ids(self, sentence, vocabulary):
        tokens = self.tokenize(sentence)
        return [vocabulary.get(w, self.UNK_ID) for w in tokens]

    def create_token_ids(self, ids_path="", vocabulary_path="", progress_interval=100000):
        if ids_path:
            self.ids_path = ids_path
        if vocabulary_path:
            self.vocabulary_path = vocabulary_path
        if gfile.Exists(self.ids_path):
            print("ids file already exists.")
            return 0
        else:
            print("Tokenizing data at %s" % self.data_path)

        vocab, _ = self.load_vocabulary(self.vocabulary_path)
        with gfile.GFile(self.data_path, mode="rb") as data_file:
            with gfile.GFile(self.ids_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % progress_interval == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = self.sentence_to_token_ids(line, vocab)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

    def create_vocab_vectors(self, vecs_path="", vocabulary_path="", vector_data_vocab_path=""):
        if not self.vector_data_path:
            print("vector data is not specified")
            return 0

        if vecs_path:
            self.vecs_path = vecs_path
        if vocabulary_path:
            self.vocabulary_path = vocabulary_path
        if gfile.Exists(self.vecs_path):
            print("vecs file already exists")
            return 0
        else:
            print("make dictionary to get vector by extracting from %s" % self.vector_data_path)

        vectors = self.read_vector_data(self.vector_data_path, vector_data_vocab_path)
        registered = 0
        vector_length = len(list(vectors.values())[0])
        _, rev_vocab = self.load_vocabulary(self.vocabulary_path)
        with gfile.GFile(self.vecs_path, mode="w") as vecs_file:
            for i, v in enumerate(rev_vocab):
                vec = []
                if i < len(self._START_VOCAB):
                    vec = ["1.0" if i == j else "0.0" for j in range(vector_length)]
                    registered += 1
                else:
                    if v in vectors:
                        vec = vectors[v]
                        registered += 1
                    else:
                        vec = ["1.0" if i == Loader.UNK_ID else "0.0" for i in range(vector_length)]

                vecs_file.write(" ".join(vec) + "\n")

        cover_rate = registered / len(rev_vocab)
        print("done to write vector file. vocab to vector cover rate is {0}".format(cover_rate))
        return cover_rate

    @classmethod
    def read_vector_data(cls, vector_data_path, vocabulary_path="", verbose=False):
        if not os.path.exists(vector_data_path):
            raise Exception("vector file is not exist.")

        rev_vec_vocab = []
        vp = vocabulary_path
        if not vp:
            default_path, ext = os.path.splitext(vector_data_path)
            if os.path.isfile(default_path + ".vocab"):
                vp = default_path + ".vocab"

        if vp:
            _, rev_vec_vocab = cls.read_vocabulary(vp)
        
        vectors = {}
        count = 0
        read_error_count = 0
        with gfile.GFile(vector_data_path, mode="rb") as dict_file:
            for i, line in enumerate(dict_file):
                count += 1
                try: 
                    els = line.strip().split()
                    word = els[0] if len(rev_vec_vocab) == 0 else rev_vec_vocab[int(els[0].decode("utf-8"))]
                    vec = [e.decode("utf-8") for e in els[1:]]
                    if not word in vectors:
                        vectors[word] = vec
                except ValueError:
                    read_error_count += 1
                    continue
                except UnicodeDecodeError:
                    read_error_count += 1
                    continue
        
        if verbose:
            print("reading error rate is {0} (total line count = {1}).".format(read_error_count / count, count))

        return vectors
    
    def load_vocab_vectors(self, vecs_path=""):
        _vp = vecs_path if vecs_path else self.vecs_path
        return self.read_vocab_vectors(_vp)

    @classmethod
    def read_vocab_vectors(self, vecs_path):
        vectors = []
        with gfile.GFile(vecs_path, mode="rb") as vec_file:
            for vec in vec_file:
                v = vec.strip().split()
                v = [float(e) for e in v]
                vectors.append(v)
        
        return vectors
