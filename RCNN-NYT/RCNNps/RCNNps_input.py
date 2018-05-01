import numpy as np
import os
import random
import _option as OPTION


class DataSet(object):
    def __init__(self, sequences_list, labels_list, shuffled=True,
                 one_hot=True, label_used=True):
        """ Construct a DataSet.
        sequences_list: [str1,str2,...], 1D list of string
        labels_list: 1D list of int

        """
        data = np.zeros([len(sequences_list), OPTION.SEQUENCE_LEN], dtype=int)
        context_left = np.zeros([len(sequences_list), OPTION.SEQUENCE_LEN], dtype=int)
        context_right = np.zeros([len(sequences_list), OPTION.SEQUENCE_LEN], dtype=int)
        for index, item in enumerate(sequences_list):
            seq_max_length = min(len(item), OPTION.SEQUENCE_LEN)
            for offset, token_id in enumerate(item[0:seq_max_length]):
                data[index][offset] = token_id
                if offset == 0:
                    context_left[index][0] = token_id
                    context_left[index][1] = token_id
                elif offset == seq_max_length - 1:
                    context_right[index][offset - 1] = token_id
                    context_right[index][offset] = token_id
                else:
                    context_left[index][offset + 1] = token_id
                    context_right[index][offset - 1] = token_id
        context_right = np.flip(context_right, 1)

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

        # Shuffle the data
        if shuffled:
            perm = np.arange(self._num_examples)
            random.shuffle(perm)
            self._data = data[perm]
            self._context_left = context_left[perm]
            self._context_right = context_right[perm]
            if label_used:
                self._labels = labels[perm]
        else:
            self._data = data
            self._context_left = context_left
            self._context_right = context_right
            if label_used:
                self._labels = labels

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
                self._context_left = self._context_left[perm]
                self._context_right = self._context_right[perm]
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
                    self._context_left = self._context_left[perm]
                    self._context_right = self._context_right[perm]
                    if self._label_used:
                        self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                self._index_in_epoch = self._num_examples
        end = self._index_in_epoch

        batch_data = self._data[start:end]
        batch_context_left = self._context_left[start:end]
        batch_context_right = self._context_right[start:end]
        if self._label_used:
            batch_labels = self._labels[start:end]

        if self._label_used:
            return batch_data, batch_context_left, batch_context_right, batch_labels
        else:
            return batch_data, batch_context_left, batch_context_right


def generate_data_set(time_list, data_dir=OPTION.DATA_PATH, data_name=OPTION.TRAIN_DATA_NAME, shuffled=True,
                      one_hot=True, label_used=True):
    """ get train data

    """
    sequences_list = []
    labels_list = []
    for t in time_list:
        lineslist = open(os.path.join(data_dir, data_name + '_%d' % t), 'r').readlines()
        count = 0
        for index, item in enumerate(lineslist):
            if index % 2 == 0:
                count = count + 1
                sequences = []
                for sens in item.strip().split('\t'):
                    if len(sequences) > OPTION.SEQUENCE_LEN:
                        break
                    sequences.extend(sens.strip().split(' '))
                sequences_list.append([int(v) for v in sequences])
            else:
                labels_list.append(int(item.strip()))

        print('time %d: %d' % (t, count))

    print('generating train data...')

    return DataSet(sequences_list, labels_list, shuffled=shuffled,
                   one_hot=one_hot, label_used=label_used)
