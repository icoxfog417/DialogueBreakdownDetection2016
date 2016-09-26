import unittest
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from model.dataset.dbd_reader import DbdReader, DbdData


class TestDbdLoader(unittest.TestCase):
    Reader = None
    
    @classmethod
    def setUpClass(cls):
        # development data for last year
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dialog_log/2015/development"))
        target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_dbd_reader.txt"))

        cls.Reader = DbdReader(data_folder, target_path, clear_when_exit=False)

    @classmethod
    def tearDownClass(cls):
        cls.Reader.remove_files()
        cls.Reader = None

    def test_get_dataset(self):
        with self.Reader as r:
            dataset, user_vocab, system_vocab = r.get_dataset()
            self.assertTrue(len(dataset) > 0)
            self.assertTrue(len(dataset[0]) == 2)  # because user/system pair

    def test_get_labels(self):
        with self.Reader as r:
            labels = r.get_labels()
            self.assertTrue(len(labels) > 0)

            # show same sample
            print("\n".join([str(labels[random.randint(0, len(labels))]) for i in range(3)]))
            for sample in labels:
                self.assertTrue(isinstance(sample.label, int))
                self.assertTrue(isinstance(sample.get_label_text(), str))
                self.assertTrue(isinstance(sample.o_prob, float))
                self.assertTrue(isinstance(sample.t_prob, float))
                self.assertTrue(isinstance(sample.x_prob, float))
                self.assertTrue(isinstance(sample.un_grammer_prob, float))

    def test_get_prob_vector(self):
        d = DbdData("user", "system", "O", 0.4, 0.5, 0.1, 1)
        v = d.get_prob_vector()
        names = DbdReader.get_label_names()

        for i, n in enumerate(names):
            if i == 0:
                self.assertEqual("X", n)
                self.assertEqual(v[i], d.x_prob)
            elif i == 1:
                self.assertEqual("O", n)
                self.assertEqual(v[i], d.o_prob)
            elif i == 2:
                self.assertEqual("T", n)
                self.assertEqual(v[i], d.t_prob)
    
    def test_ignore_t(self):
        d = DbdData("user", "system", "O", 0.6, 0.3, 0.1, 1)
        v = d.get_prob_vector(ignore_t=True)
        self.assertTrue(v[0], 0.4)
        self.assertTrue(v[1], 0.6)

        v_01 = d.get_one_hot_vector(ignore_t=True)
        self.assertEqual(2, len(v_01))
        self.assertEqual([0 , 1], v_01)

if __name__ == "__main__":
    unittest.main()
