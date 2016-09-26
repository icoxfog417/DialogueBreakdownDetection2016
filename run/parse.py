import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.dataset.loader import Loader
from model.dataset.tokenizer import JapaneseTokenizer

TOKENIZER = JapaneseTokenizer()

def make_vocab(file_path, max_vocabulary_size):
    if not os.path.isfile(file_path):
        raise Exception("Document file does not exist at {0}".format(file_path))
    
    loader = Loader(file_path, tokenizer=TOKENIZER)
    loader.create_vocabulary(max_vocabulary_size)

def make_ids_file(file_path, vocab_path):
    if not os.path.isfile(file_path):
        raise Exception("Document file does not exist at {0}".format(file_path))

    if vocab_path and os.path.isfile(vocab_path):
        loader = Loader(file_path, vocabulary_path=vocab_path, tokenizer=TOKENIZER)
    else:
        loader = Loader(file_path, tokenizer=TOKENIZER)

    loader.create_token_ids()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="document parser")
    parser.add_argument("--doc", "-d", dest="document_path", action="store", required=True, help="path to document file")
    parser.add_argument("--max_vocab", action="store", help="maximum vocabulary size", default=300000)
    parser.add_argument("--vocab", "-v", dest="vocab_path", action="store", help="path to vocabulary file (if it is set, document will be converted to id file).", default="")
    args = parser.parse_args()

    if args.document_path and args.vocab_path:
        make_ids_file(args.document_path, args.vocab_path)
    else:
        make_vocab(args.document_path, args.max_vocab)
