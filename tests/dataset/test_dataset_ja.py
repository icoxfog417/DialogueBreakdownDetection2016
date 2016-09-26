# -*- coding: utf-8 -*-
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from model.dataset.loader import Loader
from model.dataset.tokenizer import JapaneseTokenizer


class TestDataset(unittest.TestCase):
    Loader = None

    @classmethod
    def setUpClass(cls):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_data_ja.txt"))
        # every time write
        sentences = [
            "排気ガスの森",
            "あるワケナイ不条理",
            "絵に描いたような星に祈って 目醒めて",
            "誰か悲しみ　くれたら動けるかな",
            "曖昧な感情で　正体不明な感覚で今",
            "聞かせて君の夜を",
            "数え切れぬ騒ぎ出す思い",
            "長雨に濡れて",
            "お前は森になった"
        ]

        with open(data_path, "wb") as f:
            for ly in sentences:
                f.write((ly + "\n").encode("utf-8"))
        
        cls.Loader = Loader(data_path, tokenizer=JapaneseTokenizer())

    @classmethod
    def tearDownClass(cls):
        cls.Loader.remove_files()
        os.remove(cls.Loader.data_path)
        cls.Loader = None

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


if __name__ == "__main__":
    unittest.main()
