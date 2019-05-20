import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

    <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
        - embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
    """

    def __init__(self, config):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, config.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, config.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.lr = tf.Variable(float(config.learning_rate), trainable=False)
        self.lr_decay = self.lr.assign(self.lr * config.learning_rate_decay_factor)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        """
        <Variable>
            - W: 각 단어의 임베디드 벡터의 성분을 랜덤하게 할당
        """
        with tf.name_scope("embedding"):
            self.We = tf.Variable(
                tf.random_uniform([config.vocab_size, config.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.We, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        self.h_outputs = []
        self.pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, config.embedding_size, 1, config.num_filters]
                pad_input = tf.pad(self.embedded_chars_expanded,
                                   [[0, 0], [filter_size - 1, filter_size - 1], [0, 0], [0, 0]], mode='CONSTANT')
                self.Wc = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                self.bc = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    pad_input,
                    self.Wc,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                self.h = tf.nn.relu(tf.nn.bias_add(conv, self.bc), name="relu")
                # Average pooling over the outputs
                pooled = tf.nn.avg_pool(
                    self.h,
                    ksize=[1, (config.sequence_length + filter_size - 1), 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                self.h_outputs.append(self.h)
                self.pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = config.num_filters * len(config.filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.finW = tf.get_variable(
                "finW",
                shape=[num_filters_total, config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.constant(0.0, shape=[config.num_classes], name="b")
            l2_loss += tf.nn.l2_loss(self.finW)
            self.scores = tf.nn.xw_plus_b(self.h_drop, self.finW, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + config.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.label = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    # define train step
    def step(self, sess, config, x_batch, y_batch, is_training):
        if is_training:
            input_feed = {
                self.input_x: x_batch,
                self.input_y: y_batch,
                self.dropout_keep_prob: config.dropout_keep_prob
            }
            output_feed = [self.train_op, self.global_step, self.loss, self.accuracy]
            _, step, loss, accuracy = sess.run(output_feed, input_feed)
            return loss
        else:
            input_feed = {
                self.input_x: x_batch,
                self.input_y: y_batch,
                self.dropout_keep_prob: 1.0
            }
            output_feed = [self.h_outputs, self.predictions]
            actmaps, predictions = sess.run(output_feed, input_feed)
            return actmaps, predictions