
"""
Author: He Yu
Time: 2017/6/21
Description:
It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification]
(http://arxiv.org/abs/1408.5882) paper in Tensorflow.
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
"""

import tensorflow as tf
import _TF_utils as myTF
import _option as OPTION
import numpy as np
import os


class Model(object):
    """
    A CNN model for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, context_size, feature_size,
                 Word2vec = True, Trainable = False):
        """
        :param sequence_length: int. Tokens number for each input sample.Remember that we padded all our samples
                                to have the same length
        :param num_classes: int. Total classes/labels for classification.
        :param vocab_size: int. Total tokens/words number for whole dataset.This is needed to define the size of
                           our embedding layer, which will have shape [vocabulary_size, embedding_size]
        :param embedding_size: int. word2vec dimension.
        :param filter_sizes: list. The number of words we want our convolutional filters to cover.
                             We will have num_filters for each size specified here.
                             For example, [3, 4, 5] means that we will have filters that slide
                             over 3, 4 and 5 words respectively, for a total of 3 * num_filters filters.
        :param num_filters: int. The number of filters per filter size.
        """

        self._sequence_length = sequence_length
        self._num_classes = num_classes
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._context_size = context_size
        self._feature_size = feature_size

        # Embedding layer
        with tf.name_scope("embedding"):
            if Word2vec:
                lineslist = open(os.path.join(OPTION.DATA_PATH, OPTION.DATA_VEC_NAME), 'r').readlines()
                headline = lineslist[0]
                headline = [int(i) for i in (headline.strip()).split(' ')]
                self._vocab_size = headline[0] + 1  # index 0 is for padding
                self._embedding_size = headline[1]
                if embedding_size is not None:
                    assert headline[1] == embedding_size, 'error, %d!=%d'%(headline[1], embedding_size)
                vectors = np.zeros([self._vocab_size, self._embedding_size],dtype = np.float32)
                for index in range(1,self._vocab_size):
                    line = lineslist[index]
                    vec = [float(i) for i in (line.strip()).split(' ')[1:]]
                    vectors[index] = np.array(vec, dtype = np.float32)
                self.vectors = myTF.variable_with_weight_decay('vectors', vectors,
                                                               trainable = Trainable,
                                                               wd = None,
                                                               dtype = tf.float32)
            else:
                self.vectors = myTF.variable_with_weight_decay('vectors',
                                                               tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                                               trainable = Trainable,
                                                               wd = None,
                                                               dtype=tf.float32)

    def _LSTM(self, batch_size, keep_prob, inputs, scope_name):
        with tf.variable_scope(scope_name):
            # add LSTM cell
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._context_size,
                                                     forget_bias=1.0,
                                                     state_is_tuple=True)
            # add dropout layer
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,
                                                      input_keep_prob=1.0,
                                                      output_keep_prob=keep_prob)
            # initial state
            init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

            # outputs: [batch_size, time_steps, context_size]
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=inputs, initial_state=init_state,
                                               time_major=False)
        return outputs

    def inference(self, words_seq,left_context, right_context, batch_size, keep_prob):
        """
        :param words_seq: 2D tensor of [None, sequence_length]
        :return scores: 2D tensor of [None, num_classes]
        """
        # Embedding layer
        with tf.name_scope("embedding"):
            # with shape [None, sequence_length, embedding_size]
            words_embedded = tf.nn.embedding_lookup(self.vectors, words_seq)
            cl_embedded = tf.nn.embedding_lookup(self.vectors, left_context)
            cr_embedded = tf.nn.embedding_lookup(self.vectors, right_context)

        ## bulid LSTM layer
        # outputs: [batch_size, time_steps/sequence_length, context_size]
        left_reps = self._LSTM(batch_size, keep_prob, cl_embedded,'forward_LSTM')
        right_reps = self._LSTM(batch_size, keep_prob, cr_embedded,'reverse_LSTM')
        right_reps = tf.reverse(right_reps,axis=[1])

        outputs = tf.concat((left_reps,words_embedded,right_reps),axis=-1)
        # outputs = tf.concat((cl_embedded,words_embedded,cr_embedded),axis=-1)
        # with shape [batch_size, sequence_length, embedding_size, 1]
        outputs = tf.expand_dims(outputs, -1)

        with tf.name_scope("convolution"):
            # Convolution Layer
            inputmaps = self._context_size * 2 + self._embedding_size
            outputmaps = self._feature_size
            filter_shape = [1, inputmaps, 1, outputmaps]
            weights = myTF.variable_with_weight_decay('weights',
                                                      tf.truncated_normal(filter_shape, stddev=0.1),
                                                      wd=None, dtype = tf.float32)
            biases = myTF.variable_with_weight_decay('biases',
                                                     tf.constant(0.1, shape=[outputmaps]),
                                                     wd=None, dtype = tf.float32)
            # with shape [batch_size, sequence_length, feature_size]
            conv = tf.nn.conv2d(outputs,
                                weights,
                                strides=[1, 1, 1, 1],
                                padding="VALID", # no padding
                                name="conv")
            # Apply nonlinearity
            features_conv = tf.nn.relu(tf.nn.bias_add(conv, biases), name="features_conv")

        with tf.name_scope("max-pooling"):
            # Maxpooling over the outputs
            features_pooled = tf.nn.max_pool(
                features_conv,
                ksize=[1, self._sequence_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

        features = tf.reshape(features_pooled, (-1,self._feature_size))

        with tf.name_scope("dropout"):
            # Add dropout
            features_dropout =  tf.nn.dropout(features, keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            weights = myTF.variable_with_weight_decay('weights',
                                                      tf.truncated_normal([self._feature_size, self._num_classes], stddev=0.1),
                                                      wd=None, dtype = tf.float32)
            biases = myTF.variable_with_weight_decay('biases',
                                                     tf.constant(0.1, shape=[self._num_classes]),
                                                     wd=None, dtype = tf.float32)
            scores = tf.nn.xw_plus_b(features_dropout, weights, biases, name="scores")

        return scores, features



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













