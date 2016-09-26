import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from model.dataset.csv_writer import CSVWriter


class TestCSVWriter(unittest.TestCase):
    Writer = None
    
    @classmethod
    def setUpClass(cls):
        # development data for last year
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dialog_log/2015/development"))
        target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_csv.txt"))

        cls.Writer = CSVWriter(data_folder, target_path)

    @classmethod
    def tearDownClass(cls):
        cls.Writer.remove_files()
        cls.Writer = None

    def test_create_data(self):
        self.Writer.create_data()

    def test_create_loaders(self):
        self.Writer.create_data()
        user_loader, bot_loader = self.Writer.create_loaders(100)
        user_loader.remove_files()
        bot_loader.remove_files()


if __name__ == "__main__":
    unittest.main()
