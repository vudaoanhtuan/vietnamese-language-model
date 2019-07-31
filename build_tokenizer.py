import os
import pickle
import argparse

from data_loader import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word_list_file')
    parser.add_argument('--save_path', default='tokenizer.pkl')
    args = parser.parse_args()

    if not os.path.isfile(args.word_list_file):
        raise Exception("Invalid path")
    print("Building tokenzier...")
    tokenizer = build_tokenizer_from_file(args.word_list_file)

    with open(args.save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved at %s" % args.save_path)