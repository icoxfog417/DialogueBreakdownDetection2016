import os
import sys
from tensorflow.python.platform import gfile


class Reader():

    def __init__(self, path, max_size=None):
        self.path = path
        self.max_size = max_size

    def read_tokenids(self, progress_interval=100000):
        with gfile.GFile(self.path, mode="r") as f:
            line = f.readline()
            counter = 0
            while line and (not self.max_size or counter < self.max_size):
                counter += 1
                if progress_interval > 0 and counter % progress_interval == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                ids = [int(x) for x in line.strip().split()]
                line = f.readline()
                yield ids

    def read_to_list(self, progress_interval=100000):
        dataset = []
        for line in self.read_tokenids(progress_interval):
            dataset.append(line)
            
        return dataset


class ParallelReader():

    def __init__(self, path_left, path_right, max_size=None):
        self.path_left =path_left
        self.path_right = path_right
        self.max_size = max_size

    def read_pair(self, progress_interval=100000):
        left = Reader(self.path_left)
        right = Reader(self.path_right)

        counter = 0
        for l_ids, r_ids in zip(left.read_tokenids(progress_interval=-1), right.read_tokenids(progress_interval=-1)):
            counter += 1
            if counter % progress_interval == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            yield l_ids, r_ids
    
    def read_to_list(self, progress_interval=100000):
        dataset = []
        for left, right in self.read_pair(progress_interval):
            dataset.append((left, right))  # do not check whether has value (because blank utterance exists)

        return dataset
