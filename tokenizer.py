import re
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class VNTokenizer:
    def __init__(self, word_list):
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        word_list = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + word_list
        self.w2i = {w:i for i,w in enumerate(word_list)}
        self.i2w = {i:w for i,w in enumerate(word_list)}
    
    def preprocess_sentence(self, sent):
        # lower
        sent = sent.lower()
        # remove punctuation
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        # remove number
        sent = re.sub(r'[0-9]+', '', sent)
        # remove multiple white space
        sent = re.sub(r'\s+', ' ', sent)
        return sent

    def texts_to_sequences(self, list_sents, add_bos=False, add_eos=False):
        list_sents = [self.preprocess_sentence(x) for x in list_sents]
        results = []
        for sent in list_sents:
            tokens = sent.split()
            seq = [self.w2i[t] if t in self.w2i else self.w2i[self.unk_token] for t in tokens]
            if add_bos:
                seq = [self.w2i[self.bos_token]] + seq
            if add_eos:
                seq.append(self.w2i[self.eos_token])
            results.append(seq)
        return results

    def sequences_to_texts(self, sequences):
        results = []
        for seq in sequences:
            tokens = [self.i2w[t] for t in seq]
            sent = ' '.join(tokens)
            results.append(sent)
        return results
    
    def pad_sequences(self, sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=None):
        if value is None:
            value = self.w2i[self.pad_token]
        return pad_sequences(sequences, maxlen, dtype, padding, truncating, value)

