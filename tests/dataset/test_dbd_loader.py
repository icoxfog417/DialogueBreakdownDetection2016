import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from model.dataset.dbd_loader import DbdLoader


class TestDbdLoader(unittest.TestCase):
    Loader = None
    
    @classmethod
    def setUpClass(cls):
        # development data for last year
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dialog_log/2015/development"))
        target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_dbd.txt"))

        cls.Loader = DbdLoader(data_folder, target_path)

    @classmethod
    def tearDownClass(cls):
        cls.Loader.remove_files()
        cls.Loader = None

    def test_create_data(self):
        self.Loader.create_data()

    def test_create_data_by_valid_filter(self):
        self.Loader.create_data(filter='valid')

    def test_create_data_by_invalid_filter(self):
        self.Loader.create_data(filter='invalid')

    def test_create_loaders(self):
        self.Loader.create_data()
        user_loader, bot_loader = self.Loader.create_loaders(100)
        user_loader.remove_files()
        bot_loader.remove_files()


if __name__ == "__main__":
    unittest.main()
