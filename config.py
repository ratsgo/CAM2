import os
import tensorflow as tf

class Config:

    def __init__(self, root_path="./"):

        self.PAD_ID = 0
        self.UNK_ID = 1
        self.root_path = root_path
        self.data_directory = self.root_path + 'data'
        self.data_path = self.data_directory + '/ratings.txt'
        self.tokenized_path = self.data_directory + '/ratings-tokenized.txt'
        self.vocab_path = self.data_directory + '/vocab.in'
        self.id_path = self.data_directory + '/train_ids.in'
        self.summary_path = self.data_directory + '/data_summary.in'
        self.checkpoint_path = self.root_path + "checkpoint/"
        self.graph_filename = "graph.pb"
        self.vocab_cut = 1
        self.data_type = tf.float32
        self.perplexity_cut = 0.1
        self.num_classes = 2
        self.filter_sizes = [3,4,5]
        self.num_filters = 128
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 0.1
        self.batch_size = 128
        self.num_epochs = 3
        self.evaluate_every = 100
        self.checkpoint_every = 100
        self.num_checkpoints = 5
        self.learning_rate = 0.0005
        self.learning_rate_decay_factor = 0.9995
        self.max_grad_norm = 5.0
        self.max_epoch = 200000
        self.embedding_size = 128
        self.sequence_length = 20
        self.ratio = 0.3
        self.pattern = r"""[a-zA-Z0-9\`\!@#\$%\^\&\*\(\)_\-\+=\{\}\[\]:;.,\\\"'\?/~]+"""

        if os.path.exists(self.vocab_path):
            finalToken = open(self.vocab_path, "r", encoding="utf-8").readlines()[-1]
            try:
                self.vocab_size = int(finalToken.replace('\n', '').split('\u241E')[1]) + 1
            except:
                self.vocab_size = 1
        else:
            self.vocab_size = 1
