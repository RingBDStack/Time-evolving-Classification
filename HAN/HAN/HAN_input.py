
import numpy as np
import os
import random
import _option as OPTION




class DataSet(object):
    def __init__(self,sequences_list,labels_list, shuffled = True,
                 one_hot = True, label_used = True, Word2vec = True):
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

        if Word2vec:
            token_index_dict = {}
            lineslist = open(os.path.join(OPTION.DATA_PATH, OPTION.DATA_VEC_NAME), 'r').readlines()
            for index,item in enumerate(lineslist[1:]):
                token_index_dict[item.strip().split(' ')[0]] = index + 1
            data = separate_sentences(sequences_list, token_index_dict) # [#sequence, #sent, #words]

        else:
            pass

        if one_hot:
            labels = np.zeros([len(labels_list),OPTION.NUM_CLASSES],dtype=int)
            for index, item in enumerate(labels_list):
                labels[index][item] = 1
        else:
            labels = np.array(labels_list,dtype=int)

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
            if label_used:
                self._labels = labels[perm]
        else:
            self._data = data
            if label_used:
                self._labels = labels

    def get_dataset_size(self):
        return self._num_examples

    def next_batch(self, batch_size):
        """ Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        end = min(start + batch_size, self._num_examples)
        batch_data = self._data[start:end]
        if self._label_used:
            batch_labels = self._labels[start:end]

        if end == self._num_examples:
            self._epochs_completed += 1
            self._index_in_epoch = 0
            if self._shuffled:
                perm = np.arange(self._num_examples)
                random.shuffle(perm)
                self._data = self._data[perm]
                if self._label_used:
                    self._labels = self._labels[perm]
        else:
            self._index_in_epoch = end

        if self._label_used:
            return batch_data,batch_labels
        else:
            return batch_data



def generate_train_data(time_list, data_dir = OPTION.DATA_PATH, shuffled=True,
                        one_hot=True, label_used=True, Word2vec=True):
    """ get train data

    """
    sequences_list = []
    labels_list = []
    for t in time_list:
        lineslist = open(os.path.join(data_dir, OPTION.TRAIN_DATA_NAME + '_%d'%t), 'r').readlines()
        count = 0
        for index, item in enumerate(lineslist):
            if index % 2 == 0:
                count = count + 1
                sequences_list.append(item.strip())
            else:
                labels_list.append(int(item.strip()))

        print('time %d: %d' % (t,count))

    print('generating train data...')

    return DataSet(sequences_list,labels_list,shuffled = shuffled,
                 one_hot = one_hot, label_used = label_used, Word2vec = Word2vec )


def generate_eval_data(time_list, data_dir = OPTION.DATA_PATH, shuffled=False,
                        one_hot=True, label_used=True, Word2vec=True):
    sequences_list = []
    labels_list = []
    for t in time_list:
        lineslist = open(os.path.join(data_dir, OPTION.TEST_DATA_NAME + '_%d'%t),'r').readlines()
        count = 0
        for index, item in enumerate(lineslist):
            if index % 2 == 0:
                count = count + 1
                sequences_list.append(item.strip())
            else:
                labels_list.append(int(item.strip()))

        print('time %d: %d' % (t,count))

    print('generating test data...')

    return DataSet(sequences_list,labels_list,shuffled = shuffled,
                 one_hot = one_hot, label_used = label_used, Word2vec = Word2vec)

if __name__ == '__main__':
    generate_train_data(range(12))
    generate_eval_data(range(12))