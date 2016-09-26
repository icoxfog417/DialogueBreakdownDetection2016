# -*- coding: utf-8 -*-
import unittest
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from model.dataset.loader import Loader
from model.dataset.reader import Reader, ParallelReader


class TestDataset(unittest.TestCase):
    Loader = None
    VecLoader = None
    Root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_dataset"))

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.Root):
            os.mkdir(cls.Root)
        get_path = lambda name: os.path.join(cls.Root, name)
        data_path = get_path("test_data.txt")
        vector_path = get_path("test_vocab.vec")
        id_vector_path = get_path("test_vocab_ids.vec")
        id_vector_vocab_path = get_path("test_vocab_vector_dic")

        # create sample data
        sentences = [
            "Shout. nothing will begin 012",
            "Only if you act the future will come by",
            "Truth. There is only one",
            "Even if the scene is gone",
            "I can listen to my blood",
            "The intention",
            "And those crystal eyes will defend us in this fragile time",
            "Can you tell me",
            "If you feel inspired by my sound"
        ]

        with open(data_path, "wb") as f:
            for ly in sentences:
                f.write((ly + "\n").encode("utf-8"))
        
        cls.Loader = Loader(data_path, vector_data_path=vector_path)

        vocab_vectors = {}
        vec_size = 100
        for s in sentences:
            ws = cls.Loader.tokenize(s.encode("utf-8"))
            for w in [_w.decode("utf-8") for _w in ws]:
                if w not in vocab_vectors:
                    vec = ["1.0" if i == len(vocab_vectors) else "0.0" for i in range(vec_size)]
                    vocab_vectors[w] = vec
        
        with open(vector_path, "wb") as f:
            for v in vocab_vectors:
                line = v + "\t" + " ".join(vocab_vectors[v]) + "\n"
                f.write(line.encode("utf-8"))

        vocab = list(vocab_vectors.keys())
        with open(id_vector_path, "wb") as f:
            for v in vocab_vectors:
                _id = vocab.index(v)
                line = str(_id) + "\t" + " ".join(vocab_vectors[v]) + "\n"
                f.write(line.encode("utf-8"))

        with open(id_vector_vocab_path, "wb") as f:
            f.write("\n".join(vocab).encode("utf-8"))
        
        cls.VecLoader = Loader(data_path, vector_data_path=id_vector_path)
        cls.VecLoader._vocab_path = id_vector_vocab_path
        cls.VecLoader.vecs_path = get_path("test_vocab_ids_vecs")

    @classmethod
    def tearDownClass(cls):
        cls.Loader.remove_files()
        cls.VecLoader.remove_files()
        shutil.rmtree(cls.Root)

    def test_add_suffix(self):
        data_path = "c:/test/folder/data.txt"
        loader = Loader(data_path)
        suffixed = loader.add_suffix(data_path, "test")
        self.assertEqual("c:/test/folder/data_test.txt", suffixed)
    
    def test_create_and_load_vocabulary(self):
        self.Loader.create_vocabulary(20)
        vocab, rev_vocab = self.Loader.load_vocabulary()
        self.assertEqual(20, len(vocab))
    
    def test_data_to_token_ids(self):
        self.Loader.create_vocabulary(1000)
        self.Loader.create_token_ids()
        self.assertTrue(os.path.exists(self.Loader.ids_path))
        with open(self.Loader.data_path, "rb") as fd:
            with open(self.Loader.ids_path, "rb") as fn:
                read = lambda: (fd.readline(), fn.readline())
                d_line, n_line = read()
                while d_line and n_line:
                    tokens = self.Loader.tokenize(d_line)
                    numbers = n_line.split(b" ")
                    self.assertEqual(len(tokens), len(numbers))
                    d_line, n_line = read()
    
    def test_reader(self):
        self.Loader.create_vocabulary(1000)
        self.Loader.create_token_ids()
        reader = Reader(self.Loader.ids_path)

        for ids in reader.read_tokenids():
            for i in ids:
                self.assertTrue(isinstance(i, int))

    def test_parallel_reader(self):
        self.Loader.create_vocabulary(1000)
        self.Loader.create_token_ids()
        reader = ParallelReader(self.Loader.ids_path, self.Loader.ids_path)
        
        for left, right in reader.read_pair():
            self.assertEqual(left, right)
    
    def test_create_vector_vocab(self):
        self.Loader.create_vocabulary(20)
        cover_rate = self.Loader.create_vocab_vectors()
        self.assertEqual(1, cover_rate)
        vector_vocab = self.Loader.load_vocab_vectors()
        self.assertEqual(20, len(vector_vocab))

    def test_create_vector_vocab_from_id(self):
        self.VecLoader.create_vocabulary(20)
        cover_rate = self.VecLoader.create_vocab_vectors(vector_data_vocab_path=self.VecLoader._vocab_path)
        self.assertEqual(1, cover_rate)
        vector_vocab = self.Loader.load_vocab_vectors()
        self.assertEqual(20, len(vector_vocab))


if __name__ == "__main__":
    unittest.main()
