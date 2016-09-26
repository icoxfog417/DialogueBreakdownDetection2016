import os
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.externals import joblib
from model.baseline.detector import Detector
from model.dataset.loader import Loader


class ConsciousDetector(Detector):

    def __init__(self, session, model_interface, model_dir="", name="", ignore_t=False, load_if_exist=True):
        self.model_interface = model_interface
        self.model_interface.build(session, predict=False, projection=True)
        self.classifier = SVC(kernel="rbf", probability=True)
        self.ignore_t = ignore_t
        self.model_dir = model_dir
        self.name = name if name else self.__class__.__name__.lower()
        if load_if_exist and os.path.isfile(self._make_model_path()):
            self.load()

    def extract_feature(self, session, utterance_pair):
        u, s = utterance_pair
        s_trim_unk = [w for w in s if w != Loader.UNK_ID]
        output, decoder_state, encoder_state = self.model_interface.predict(session, u, s_trim_unk)
        # output: selected bucket size x batch_size(=1 for predict) x vocab_size
        # state: selected bucket size x model_size

        result = [int(np.argmax(o, axis=1)) for o in output]
        last = len(result)

        if Loader.EOS_ID in result:
            _tail = result.index(Loader.EOS_ID)
            if _tail > 0:
                last = _tail
        
        indices = ([Loader.OK_ID, Loader.NG_ID] + [] if self.ignore_t else [Loader.NEITHER_ID])
        # simple word match
        match_rate = 0 if len(s) == 0 else sum([1 for w in s_trim_unk if w in u]) / len(s)
        #relative_counts = [int(np.argmax(o[:, (Loader.OK_ID, Loader.NG_ID, Loader.NEITHER_ID)], axis=1)) for o in output]
        breakdown_x = [np.max(output[i][:, r]) if r == Loader.NG_ID else 0 for i, r in enumerate(result)]
        breakdown_o = [np.max(output[i][:, r]) if r == Loader.OK_ID else 0 for i, r in enumerate(result)]

        #breakdown_maxs_n = np.array(breakdown_maxs) - np.mean(breakdown_maxs, axis=0) / np.std(breakdown_maxs, axis=0)
        #breakdown_means = [np.mean(o[:, (Loader.OK_ID, Loader.NG_ID, Loader.NEITHER_ID)], axis=0) for o in output[:last]]
        surface_count = [result.count(Loader.OK_ID), result.count(Loader.NG_ID)] + [] if self.ignore_t else [result.count(Loader.NEITHER_ID)]
        #surface_count = [0] * len(surface_count) if len(result) == 0 else [c / len(result) for c in surface_count]
        features = []
        ## features += np.random.uniform(size=3).tolist()  # 0.39, and fixed 0 is 0.43. so below of this is sign of bug.
        
        #features += np.max(breakdown_maxs, axis=0).tolist()  # standalone acc is 0.58
        #features = np.max(breakdown_maxs_n, axis=0).tolist()
        features += surface_count  # standalone acc is 0.6
        features += [match_rate]
        features += [max(breakdown_x), max(breakdown_o)]
        #return features
        #return decoder_state[-1].tolist() + [max(breakdown_x), max(breakdown_o)]
        return decoder_state[-1].tolist() + encoder_state[-1].tolist()
