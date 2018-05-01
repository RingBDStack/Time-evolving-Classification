import numpy as np
import os
import random
import _option as OPTION
import _TF_utils as myTF


class DataSet(object):
    def __init__(self, sequences_list, labels_list, features_array=None, shuffled=True,
                 one_hot=True, label_used=True, Word2vec=True):
        """ Construct a DataSet.
        sequences_list: [str1,str2,...], 1D list of string
        labels_list: 1D list of int

        """

        def separate_sentences(lines, word_index):
            passages = []
            texts = []
            for text in lines:
                texts.append(text)
                sentences = text.split('\t')
                passages.append(sentences)
            data = np.zeros((len(texts), OPTION.SEQUENCE_LEN, OPTION.SENT_LEN), dtype='int32')
            for i, sentences in enumerate(passages):
                for j, sent in enumerate(sentences):
                    if j < OPTION.SEQUENCE_LEN:
                        wordTokens = sent.strip(' ').split(' ')
                        k = 0
                        for _, word in enumerate(wordTokens):
                            if k < OPTION.SENT_LEN:
                                data[i, j, k] = int(word)

            return data

        token_index_dict = {}
        lineslist = open(os.path.join(OPTION.DATA_PATH, OPTION.DATA_VEC_NAME), 'r').readlines()
        for index, item in enumerate(lineslist[1:]):
            token_index_dict[item.strip().split(' ')[0]] = index + 1
        data = separate_sentences(sequences_list, token_index_dict)  # [#sequence, #sent, #words]

        if one_hot:
            labels = np.zeros([len(labels_list), OPTION.NUM_CLASSES], dtype=int)
            for index, item in enumerate(labels_list):
                labels[index][item] = 1
        else:
            labels = np.array(labels_list, dtype=int)

        self._shuffled = shuffled
        self._one_hot = one_hot
        self._label_used = label_used
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(sequences_list)

        self._data = data
        if label_used:
            self._labels = labels
        self._features_array = features_array

        # Shuffle the data
        if shuffled:
            self._shuffle_dataset()

    def _shuffle_dataset(self):
        if self._shuffled:
            perm = np.arange(self._num_examples)
            random.shuffle(perm)
            self._data = self._data[perm]
            if self._features_array is not None:
                self._features_array = self._features_array[perm]
            if self._label_used:
                self._labels = self._labels[perm]

    def get_dataset_size(self):
        return self._num_examples

    def next_batch(self, batch_size, keep_strict_batching=False):
        """ Return the next `batch_size` examples from this data set."""
        if keep_strict_batching:
            assert batch_size <= self._num_examples

        if self._index_in_epoch >= self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffled:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self._data[perm]
                if self._features_array is not None:
                    self._features_array = self._features_array[perm]
                if self._label_used:
                    self._labels = self._labels[perm]
            # Start next epoch
            self._index_in_epoch = 0

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            if keep_strict_batching:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                if self._shuffled:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._data = self._data[perm]
                    if self._features_array is not None:
                        self._features_array = self._features_array[perm]
                    if self._label_used:
                        self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                self._index_in_epoch = self._num_examples
        end = self._index_in_epoch

        batch_data = self._data[start:end]
        if self._label_used:
            batch_labels = self._labels[start:end]
        if self._features_array is not None:
            batch_features = self._features_array[start:end]
        else:
            batch_features = None

        if self._label_used:
            return batch_data, batch_labels, batch_features
        else:
            return batch_data, batch_features


def generate_train_data(data_time, data_dir=OPTION.DATA_PATH, shuffled=True,
                        one_hot=True, label_used=True, Word2vec=True):
    """ get train data

    """
    sequences_list = []
    labels_list = []
    features_list = []

    start_time = max(data_time - OPTION.DP_DEPTH, 0)

    lineslist = open(os.path.join(data_dir, OPTION.TRAIN_DATA_NAME + '_%d' % data_time), 'r').readlines()
    count = 0
    for index, item in enumerate(lineslist):
        if index % 2 == 0:
            count = count + 1
            sequences_list.append(item.strip())
        else:
            labels_list.append(int(item.strip()))

    print('time %d: %d' % (data_time, count))
    # read features
    #  read features : 'features_%s_%d_%d' % (name,time,model_time)
    for model_time in range(start_time, data_time):
        features_t_time = np.loadtxt(
            os.path.join(OPTION.MODELPARA_DIR, 'features_train_%d_%d' % (data_time, model_time)),
            dtype=float)
        features_list.append(features_t_time)
    if len(features_list) > 0:
        features_array = np.concatenate(features_list, axis=1)
    else:
        features_array = None

    print('generating train data...')

    return DataSet(sequences_list, labels_list, features_array=features_array, shuffled=shuffled,
                   one_hot=one_hot, label_used=label_used, Word2vec=Word2vec)


def generate_eval_data(data_time, data_dir=OPTION.DATA_PATH, shuffled=False,
                       one_hot=True, label_used=True, Word2vec=True):
    sequences_list = []
    labels_list = []
    features_list = []

    start_time = max(data_time - OPTION.DP_DEPTH, 0)

    lineslist = open(os.path.join(data_dir, OPTION.TEST_DATA_NAME + '_%d' % data_time), 'r').readlines()
    count = 0
    for index, item in enumerate(lineslist):
        if index % 2 == 0:
            count = count + 1
            sequences_list.append(item.strip())
        else:
            labels_list.append(int(item.strip()))

    print('time %d: %d' % (data_time, count))
    # read features
    #  read features : 'features_%s_%d_%d' % (name,time,model_time)
    for model_time in range(start_time, data_time):
        features_t_time = np.loadtxt(
            os.path.join(OPTION.MODELPARA_DIR, 'features_test_%d_%d' % (data_time, model_time)), dtype=float)
        features_list.append(features_t_time)
    if len(features_list) > 0:
        features_array = np.concatenate(features_list, axis=1)
    else:
        features_array = None

    print('generating test data...')

    return DataSet(sequences_list, labels_list, features_array=features_array, shuffled=shuffled,
                   one_hot=one_hot, label_used=label_used, Word2vec=Word2vec)


def generate_feature_data(data_time, model_time, data_dir=OPTION.DATA_PATH, shuffled=True,
                          one_hot=True, label_used=True, Word2vec=True, isTrain=True):
    """ get feature data

    """
    if isTrain:
        data_name = OPTION.TRAIN_DATA_NAME
        feature_name = 'features_train'
    else:
        data_name = OPTION.TEST_DATA_NAME
        feature_name = 'features_test'

    sequences_list = []
    labels_list = []
    features_list = []
    start_time = max(model_time - OPTION.DP_DEPTH, 0)
    lineslist = open(os.path.join(data_dir, data_name + '_%d' % data_time), 'r').readlines()
    count = 0
    for index, item in enumerate(lineslist):
        if index % 2 == 0:
            count = count + 1
            sequences_list.append(item.strip())
        else:
            labels_list.append(int(item.strip()))

    print('time %d: %d' % (data_time, count))

    for model_ in range(start_time, model_time):
        features_t_time = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, '%s_%d_%d' % (feature_name, data_time, model_)),
                                     dtype=float)
        features_list.append(features_t_time)
    if len(features_list) > 0:
        features_array = np.concatenate(features_list, axis=1)
    else:
        features_array = None

    return DataSet(sequences_list, labels_list, features_array=features_array, shuffled=shuffled,
                   one_hot=one_hot, label_used=label_used, Word2vec=Word2vec)
