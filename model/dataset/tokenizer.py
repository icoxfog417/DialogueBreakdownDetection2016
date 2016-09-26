import re
import MeCab


class Tokenizer():

    def __init__(self):
        pass
    
    def tokenize(self, sentence):
        raise Exception("Tokenizer have to implements tokenize method")


class BasicTokenizer(Tokenizer):
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

    def tokenize(self, sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(re.split(self._WORD_SPLIT, space_separated_fragment))
        return [w for w in words if w]


class JapaneseTokenizer(Tokenizer):

    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")  # don't need pos data
        self.tagger.parse("")  # trick to avoid blank surface 

    def tokenize(self, sentence):
        decoded = sentence.decode("utf-8")
        words = []
        node = self.tagger.parseToNode(decoded)
        while node:
            w = node.surface.strip()
            if w:
                words.append(w.encode("utf-8"))
            node = node.next
        
        return words
    


