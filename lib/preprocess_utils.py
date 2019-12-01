import os
import re
from konlpy.tag import Mecab

class preprocessor:

    def __init__(self, config):
        self.config = config
        if not os.path.exists(self.config.data_directory):
            os.makedirs(self.config.data_directory)
        if os.path.exists(self.config.tokenized_path) \
                and os.path.exists(self.config.vocab_path):
            print("preprocessing, PASS!")
        else:
            print("data preprocessing ...")
            self.tk_list = []
            self.tk_freq = []
            self.tokenizer = Mecab()
            self.apply()
            self.save_dict()

    def apply(self):
        if not os.path.exists(self.config.tokenized_path):
            with open(self.config.data_path, "r", encoding="utf-8") as f1, \
                    open(self.config.tokenized_path, 'w', encoding='utf-8') as f2:
                for line in f1:
                    _, string, label = line.replace('\n', '').split('\t')
                    sentence = re.sub(self.config.pattern, " ", string)
                    tokens = self.tokenizer.morphs(sentence)
                    f2.writelines(' '.join(tokens) + '\u241E' + label + '\n')
        else:
            print("tokenizing, PASS!")

    def save_dict(self):
        if not os.path.exists(self.config.vocab_path):
            print("build vocab...")
            with open(self.config.tokenized_path, 'r', encoding='utf-8') as f1:
                for line in f1:
                    tokens, _ = line.strip().split("\u241E")
                    for token in tokens.split():
                        if token not in self.tk_list:
                            self.tk_list.append(token)
                            self.tk_freq.append(1)
                        else:
                            self.tk_freq[self.tk_list.index(token)] += 1
            tk_sort = sorted(list(zip(self.tk_list, self.tk_freq)), key=lambda x: x[1], reverse=True)
            vocab = ["_PAD\u241E" + str(self.config.PAD_ID), "_UNK\u241E" + str(self.config.UNK_ID)]
            vocab_dict_idx = len(vocab)
            for tk_candidate, word_freq in tk_sort:
                if word_freq >= self.config.vocab_cut:
                    vocab.append(tk_candidate + "\u241E" + str(vocab_dict_idx))
                    vocab_dict_idx += 1
            with open(self.config.vocab_path, 'w', encoding='utf-8') as f2:
                for vocab_line in vocab:
                    f2.writelines(vocab_line + "\n")
            with open(self.config.summary_path, 'w', encoding='utf-8') as f3:
                f3.writelines("vocab/%d" % (len(vocab)) + "\n")
        else:
            print("construct vocab, PASS!")


class chat_to_ids:

    def __init__(self, config):
        self.config = config
        if os.path.exists(self.config.id_path):
            print("chat to ids, PASS!")
        else:
            print("build chat to ids...")
            self.vocab_dict = {}
            with open(self.config.vocab_path) as f1:
                for line in f1:
                    token, id = line.replace('\n', '').split('\u241E')
                    self.vocab_dict[token] = str(id)
            self.apply()

    def apply(self):
        with open(self.config.tokenized_path, 'r', encoding='utf-8') as f1, \
                open(self.config.id_path, 'w', encoding='utf-8') as f2:
            for line in f1:
                tokens, label = line.replace('\n', '').split("\u241E")
                tokens = tokens.split()
                token_idx_line = []
                for token in tokens:
                    try:
                        token_idx = self.vocab_dict[token]
                    except KeyError:
                        token_idx = self.vocab_dict["_UNK"]
                    token_idx_line.append(token_idx.replace('\n', ''))
                f2.writelines(" ".join(token_idx_line) + '\u241E' + label + '\n')