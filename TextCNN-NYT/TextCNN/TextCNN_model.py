
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

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
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
        self._filter_sizes = filter_sizes
        self._num_filters = num_filters

        # Embedding layer
        with tf.name_scope("embedding"):
            if Word2vec:
                lineslist = open(os.path.join(OPTION.DATA_PATH, OPTION.DATA_VEC_NAME), 'r').readlines()
                headline = lineslist[0]
                headline = [int(i) for i in (headline.strip()).split(' ')]
                self._vocab_size = headline[0] + 1  # index 0 is for padding
                self._embedding_size = headline[1]
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

    def inference(self, input, eval_data=False):
        """
        :param input: 2D tensor of [None, sequence_length]
        :return scores: 2D tensor of [None, num_classes]
        """

        # Embedding layer
        with tf.name_scope("embedding"):
            # with shape [None, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(self.vectors, input)
            # with shape [None, sequence_length, embedding_size, 1]
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self._filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                weights = myTF.variable_with_weight_decay('weights',
                                                          tf.truncated_normal(filter_shape, stddev=0.1),
                                                          wd=None, dtype = tf.float32)
                biases = myTF.variable_with_weight_decay('biases',
                                                         tf.constant(0.1, shape=[self._num_filters]),
                                                         wd=None, dtype = tf.float32)
                conv = tf.nn.conv2d(embedded_chars_expanded,
                                    weights,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                ouput = tf.nn.relu(tf.nn.bias_add(conv, biases), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    ouput,
                    ksize=[1, self._sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self._num_filters * len(self._filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        features = h_pool_flat

        # Add dropout
        if not eval_data:
            with tf.name_scope("dropout"):
                h_pool_flat = tf.nn.dropout(h_pool_flat, OPTION.DROPOUT_KEEP_PROB)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            weights = myTF.variable_with_weight_decay('weights',
                                                      tf.truncated_normal([num_filters_total, self._num_classes], stddev=0.1),
                                                      wd=None, dtype = tf.float32)
            biases = myTF.variable_with_weight_decay('biases',
                                                     tf.constant(0.1, shape=[self._num_classes]),
                                                     wd=None, dtype = tf.float32)
            scores = tf.nn.xw_plus_b(h_pool_flat, weights, biases, name="scores")

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













