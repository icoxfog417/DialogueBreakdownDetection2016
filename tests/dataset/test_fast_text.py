# -*- coding: utf-8 -*-
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import fasttext
from model.dataset.loader import Loader
from model.dataset.dbd_reader import DbdReader


class TestFastText(unittest.TestCase):
    def get_path(self, file_name):
        return os.path.join(os.path.dirname(__file__), "../data/" + file_name)

    def xtest_read_model(self):
        path = self.get_path("model_w2v.bin")
        print("now loading model from bin file...")
        model = fasttext.load_model(path)  # now cause UnicodeDecodeError
        print("done load the model!")
        self.assertTrue(model)
    
    def xtest_read_vec(self):
        path = self.get_path("model_w2v.vec")
        vectors = Loader.read_vector_data(path)  # terrible memory use (may because 2G byte over...)
        self.assertTrue(len(vectors) > 0)
    
    def test_read_vec_with_vocab(self):
        path = self.get_path("model_tokened.vec")
        vocab = self.get_path("model_vocab")
        vectors = Loader.read_vector_data(path, vocab)  # works fine!
        self.assertTrue(len(vectors) > 0)
        
        sample_count = 5
        for i, v in enumerate(vectors):
            print("{0}: {1}...".format(v.decode("utf-8"), " ".join([str(e) for e in vectors[v][:5]])))
            if i > sample_count:
                break
        
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dialog_log/all/train"))
        target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_fast_text.txt"))

        with DbdReader(data_folder, target_path) as reader:
            reader.user_loader.vector_data_path = path
            rate = reader.user_loader.create_vocab_vectors(vector_data_vocab_path=vocab)
            print(rate)  # 0.951


if __name__ == "__main__":
    unittest.main()
