"""
Author: He Yu, He Mutian
Time: 2017/12/26
Description:
Implementation of Hierarchical Attention Network
"""

import tensorflow as tf
import _TF_utils as myTF
import _option as OPTION
import numpy as np
import os


class Model(object):
    """
    A HAN model for text classification.
    """

    def __init__(self, sequence_length, sent_length, num_classes, vocab_size, embedding_size,
                 Word2vec=True, Trainable=False):
        """
        :param sequence_length: int. Number of sentences for each input sample.Remember that we padded all our samples
                                to have the same length
        :param sent_length: int. Similar to sequence_length, number of tokens in each (padded) sentence
        :param num_classes: int. Total classes/labels for classification.
        :param vocab_size: int. Total tokens/words number for whole dataset.This is needed to define the size of
                           our embedding layer, which will have shape [vocabulary_size, embedding_size]
        :param embedding_size: int. word2vec dimension.
        """

        self._sequence_length = sequence_length
        self._sent_length = sent_length
        self._num_classes = num_classes
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size

        # Embedding layer
        with tf.name_scope("embedding"):
            if Word2vec:
                lineslist = open(os.path.join(OPTION.DATA_PATH, OPTION.DATA_VEC_NAME), 'r').readlines()
                headline = lineslist[0]
                headline = [int(i) for i in (headline.strip()).split(' ')]
                self._vocab_size = headline[0] + 1  # index 0 is for padding
                self._embedding_size = headline[1]
                if embedding_size is not None:
                    assert headline[1] == embedding_size, 'error, %d!=%d' % (headline[1], embedding_size)
                vectors = np.zeros([self._vocab_size, self._embedding_size], dtype=np.float32)
                for index in range(1, self._vocab_size):
                    line = lineslist[index]
                    vec = [float(i) for i in (line.strip()).split(' ')[1:]]
                    vectors[index] = np.array(vec, dtype=np.float32)
                self.vectors = myTF.variable_with_weight_decay('vectors', vectors,
                                                               trainable=Trainable,
                                                               wd=None,
                                                               dtype=tf.float32)
            else:
                self.vectors = myTF.variable_with_weight_decay('vectors',
                                                               tf.random_uniform([vocab_size, embedding_size], -1.0,
                                                                                 1.0),
                                                               trainable=Trainable,
                                                               wd=None,
                                                               dtype=tf.float32)

    def inference(self, input, features_before, eval_data=False):
        """
        :param
            input: 2D tensor of [None, sequence_length, sent_length]
            features_before: list, 3D tensor of [batch_size, timestep_size, feature_size]
        :return scores: 2D tensor of [None, num_classes]
        """

        # Embedding layer
        with tf.name_scope("embedding"):
            # with shape [None, sequence_length, sent_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(self.vectors, input)
            # with shape [None, sequence_length, sent_length, embedding_size, 1]
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a biRNN for each sentence
        def biRNNLayer(inputs, hidden_size, name):
            # inputs: [batch_size, n_step, dim]
            def length(sequences):
                used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
                seq_len = tf.reduce_sum(used, reduction_indices=1)
                return tf.cast(seq_len, tf.int32)

            with tf.variable_scope(name):
                GRU_cell_fw = tf.contrib.rnn.GRUCell(hidden_size)
                GRU_cell_bw = tf.contrib.rnn.GRUCell(hidden_size)
                if not eval_data:
                    GRU_cell_fw = tf.contrib.rnn.DropoutWrapper(cell=GRU_cell_fw,
                                                                input_keep_prob=1.0,
                                                                output_keep_prob=OPTION.DROPOUT_KEEP_PROB)
                    GRU_cell_bw = tf.contrib.rnn.DropoutWrapper(cell=GRU_cell_bw,
                                                                input_keep_prob=1.0,
                                                                output_keep_prob=OPTION.DROPOUT_KEEP_PROB)
                inputs_list = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
                outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=GRU_cell_fw,
                                                               cell_bw=GRU_cell_bw,
                                                               inputs=inputs_list,
                                                               sequence_length=length(inputs),
                                                               dtype=tf.float32)
                # outputs: [batch_size, n_step, hidden_size*2]
                outputs = tf.transpose(tf.stack(outputs, 0), [1, 0, 2])
                return outputs

        def AttentionLayer(inputs, n_step, hidden_size, name):
            # inputs: [batch_size, n_step, hidden_size * 2]
            with tf.variable_scope(name):
                u_context = tf.Variable(tf.truncated_normal([hidden_size * 2]), name='u_context')
                weights = myTF.variable_with_weight_decay(name + '_weights',
                                                          tf.truncated_normal([hidden_size * 2], stddev=0.1),
                                                          wd=None, dtype=tf.float32)
                biases = myTF.variable_with_weight_decay(name + '_biases',
                                                         tf.constant(0.1, shape=[hidden_size * 2]),
                                                         wd=None, dtype=tf.float32)
                # [batch_size, n_step, hidden_size * 2]
                h = tf.tanh(inputs * weights + biases)
                # [batch_size, n_step, 1]
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
                # [batch_size, hidden_size * 2]
                outputs = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
                return outputs

        with tf.name_scope("sent2vec"):
            # [batch_size * sequence_length, sent_length, embedding_size]
            word_embedded = tf.reshape(embedded_chars, [-1, self._sent_length, self._embedding_size])
            # [batch_size * sequence_length, sent_length, hidden_size*2]
            word_encoded = biRNNLayer(word_embedded, hidden_size=OPTION.WORD_HIDDEN_SIZE, name='word_encoder')
            # [batch_size * sequence_length, hidden_size*2]
            sent_vec = AttentionLayer(word_encoded, n_step=self._sent_length, hidden_size=OPTION.WORD_HIDDEN_SIZE,
                                      name='word_attention')
            sent_vec = tf.reshape(sent_vec, [-1, self._sequence_length, OPTION.WORD_HIDDEN_SIZE * 2])

        with tf.name_scope("doc2vec"):
            # [batch_size, sequence_length, hidden_size * 2]
            doc_encoded = biRNNLayer(sent_vec, OPTION.SENT_HIDDEN_SIZE, name='sent_encoder')
            # [batch_size, hidden_size*2]
            doc_vec = AttentionLayer(doc_encoded, n_step=self._sequence_length, hidden_size=OPTION.SENT_HIDDEN_SIZE,
                                     name='sent_attention')

        with tf.name_scope("feature"):
            weights = myTF.variable_with_weight_decay('weights',
                                                      tf.truncated_normal(
                                                          [OPTION.SENT_HIDDEN_SIZE * 2, OPTION.FEATURE_SIZE],
                                                          stddev=0.1),
                                                      wd=None, dtype=tf.float32)
            biases = myTF.variable_with_weight_decay('biases',
                                                     tf.constant(0.1, shape=[OPTION.FEATURE_SIZE]),
                                                     wd=None, dtype=tf.float32)
            # [batch_size, feature_size]
            feature = tf.nn.xw_plus_b(doc_vec, weights, biases, name="features")

        ret_feature = feature

        if features_before is not None:
            concat_feature = tf.concat([features_before, feature], 1)
            weights = myTF.variable_with_weight_decay('weights',
                                                      tf.truncated_normal(
                                                          [concat_feature.get_shape()[1].value, OPTION.FEATURE_SIZE],
                                                          stddev=0.1),
                                                      wd=None, dtype=tf.float32)
            biases = myTF.variable_with_weight_decay('biases',
                                                     tf.constant(0.1, shape=[OPTION.FEATURE_SIZE]),
                                                     wd=None, dtype=tf.float32)
            feature = tf.matmul(concat_feature, weights)
            feature = tf.nn.relu(tf.nn.bias_add(feature, biases))

        # Add dropout
        if not eval_data:
            with tf.name_scope("dropout"):
                feature = tf.nn.dropout(feature, OPTION.DROPOUT_KEEP_PROB)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            weights = myTF.variable_with_weight_decay('weights',
                                                      tf.truncated_normal(
                                                          [feature.get_shape()[1].value, self._num_classes],
                                                          stddev=0.1),
                                                      wd=None, dtype=tf.float32)
            biases = myTF.variable_with_weight_decay('biases',
                                                     tf.constant(0.1, shape=[self._num_classes]),
                                                     wd=None, dtype=tf.float32)
            scores = tf.nn.xw_plus_b(feature, weights, biases, name="scores")
        return scores, ret_feature


def calculate_loss(logits, labels):
    """ The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    Args:
        logits: Logits from inference(), 2D tensor of [None, NUM_CLASSES].
        labels: 2D tensor of [None, NUM_CLASSES].
    Returns:
        Loss: 0D tensor of type float.
    """
    myTF.calculate_cross_entropy_loss(logits, labels)

    return tf.add_n(tf.get_collection('losses'), name='loss_total')
