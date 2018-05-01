
import numpy as np
import os
import random
import _option as OPTION
import _TF_utils as myTF




class DataSet(object):
    def __init__(self,sequences_list,labels_list, features_array=None, shuffled = True,
                 one_hot = True, label_used = True, Word2vec = True):
        """ Construct a DataSet.
        sequences_list: [str1,str2,...], 1D list of string
        labels_list: 1D list of int
        
        """
        if Word2vec:
            token_index_dict = {}
            lineslist = open(os.path.join(OPTION.DATA_PATH, OPTION.DATA_VEC_NAME), 'r').readlines()
            for index,item in enumerate(lineslist[1:]):
                token_index_dict[item.strip().split(' ')[0]] = index + 1
            data = np.zeros([len(sequences_list), OPTION.SEQUENCE_LEN],dtype=int)
            for index,item in enumerate(sequences_list):
                tokens_list = item.split(' ')
                offset = 0
                for index_2 in range(0,len(tokens_list)):
                    if offset >= OPTION.SEQUENCE_LEN:
                        break
                    if tokens_list[index_2] in token_index_dict.keys():
                        data[index][offset] = token_index_dict[tokens_list[index_2]]
                        offset +=1
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

    def next_batch(self, batch_size):
        """ Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        end = min(start + batch_size, self._num_examples)
        batch_data = self._data[start:end]
        if self._label_used:
            batch_labels = self._labels[start:end]
        if self._features_array is not None:
            batch_features_array = self._features_array[start:end]

        if end == self._num_examples:
            self._epochs_completed += 1
            self._index_in_epoch = 0
            if self._shuffled:
                perm = np.arange(self._num_examples)
                random.shuffle(perm)
                self._data = self._data[perm]
                if self._features_array is not None:
                    self._features_array = self._features_array[perm]
                if self._label_used:
                    self._labels = self._labels[perm]
        else:
            self._index_in_epoch = end

        if self._label_used:
            if self._features_array is not None:
                return batch_data, batch_labels, batch_features_array
            else:
                return batch_data, batch_labels
        else:
            if self._features_array is not None:
                return batch_data, batch_features_array
            else:
                return batch_data



def generate_train_data(time_list, data_dir = OPTION.DATA_PATH, shuffled=True,
                        one_hot=True, label_used=True, Word2vec=True):
    """ get train data

    """
    sequences_list = []
    labels_list = []
    features_list = []
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
        # read features : 'features_%s_%d_%d' % (name,time,model_time)
        features_t_list = []
        for time in range(0,time_list[-1]):
            features_t_time = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, 'features_train_%d_%d' % (t,time)),dtype=float)
            features_t_time = features_t_time[:,np.newaxis,:]
            features_t_list.append(features_t_time)

        features_t = np.concatenate(features_t_list,axis=1)
        assert np.size(features_t,0)==count, 'error, %d != %d'%(np.size(features_t,0), count)
        features_list.append(features_t)

    features_array = np.concatenate(features_list,axis=0)

    print('generating train data...')

    return DataSet(sequences_list,labels_list, features_array=features_array,shuffled = shuffled,
                 one_hot = one_hot, label_used = label_used, Word2vec = Word2vec )


def generate_eval_data(time_list, data_dir = OPTION.DATA_PATH, shuffled=False,
                        one_hot=True, label_used=True, Word2vec=True):
    sequences_list = []
    labels_list = []
    # features_list = []
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
        # # read features
        # features_t_list = []
        # for time in range(0, time_list[-1]):
        #     features_t_time = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, 'features_test_%d_%d' % (t, time)), dtype=float)
        #     features_t_time = features_t_time[:, np.newaxis, :]
        #     features_t_list.append(features_t_time)
        #     # print(np.size(features_t_time))
        # features_t = np.concatenate(features_t_list, axis=1)
        # assert np.size(features_t, 0) == count, 'error, %d != %d' % (np.size(features_t, 0), count)
        # features_list.append(features_t)

    # features_array = np.concatenate(features_list, axis=0)
    print('generating test data...')

    return DataSet(sequences_list,labels_list,features_array=None,shuffled = shuffled,
                 one_hot = one_hot, label_used = label_used, Word2vec = Word2vec)




def generate_feature_data(time_list,model_time, data_dir = OPTION.DATA_PATH, shuffled=True,
                        one_hot=True, label_used=True, Word2vec=True, isTrain = True):
    """ get feature data

    """
    if isTrain:
        data_name = OPTION.TRAIN_DATA_NAME
        feature_name = 'features_train'
    else:
        data_name =  OPTION.TEST_DATA_NAME
        feature_name = 'features_test'

    sequences_list = []
    labels_list = []
    # features_list = []
    for t in time_list:
        lineslist = open(os.path.join(data_dir, data_name + '_%d'%t), 'r').readlines()
        count = 0
        for index, item in enumerate(lineslist):
            if index % 2 == 0:
                count = count + 1
                sequences_list.append(item.strip())
            else:
                labels_list.append(int(item.strip()))

        print('time %d: %d' % (t,count))
        # # read features : 'features_%s_%d_%d' % (name,time,model_time)
        # features_t_list = []
        # for time in range(0,model_time):
        #     features_t_time = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, '%s_%d_%d' % (feature_name, t,time)),dtype=float)
        #     features_t_time = features_t_time[:,np.newaxis,:]
        #     features_t_list.append(features_t_time)
        #
        # features_t = np.concatenate(features_t_list,axis=1)
        # assert np.size(features_t,0)==count, 'error, %d != %d'%(np.size(features_t,0), count)
        # features_list.append(features_t)

    # features_array = np.concatenate(features_list,axis=0)

    return DataSet(sequences_list,labels_list, features_array=None,shuffled = shuffled,
                 one_hot = one_hot, label_used = label_used, Word2vec = Word2vec )

