import os

## path
DATA_PATH = '../data/NYT' # Path to data directory
TRAIN_DIR = './tmp/train'  # Directory where to write event logs and checkpoint.
EVAL_DIR = './tmp/eval'
MODELPARA_DIR = './tmp'
CHECKPOINT_DIR = './tmp/train'
PRE_TRAIN_MODEL = './tmp/pre_train_model'

## file
TRAIN_DATA_NAME = 'train_data'
TEST_DATA_NAME = 'test_data'
DATA_OPTION_NAME = 'option'
DATA_VEC_NAME = 'word2vec_NYT.vectors'

MEMORY_ALLOW_GROWTH = False


## model parameters
DROPOUT_KEEP_PROB = 0.5  # Add a dropout during training.
INITIAL_LEARNING_RATE = 0.0001 # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.999 # The decay to use for the moving average.
WEIGHT_DECAY = 0.0005     # 0.00005  # 0.0005 # l2 regularization weight decay


BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
EMEBEDDING_DIMENSION = 100
MAX_CKPT_PERIOD = 500 #1000 #3000 # Dimensionality of character embedding
NUM_EPOCHS = 300  # Number of epochs to run.
MIN_CKPTS = 1
SUMMARY_PERIOD = 30

NUM_CLASSES = 26
SEQUENCE_LEN = 64 
SENT_LEN = 64

WORD_HIDDEN_SIZE = 128
SENT_HIDDEN_SIZE = 128

PARAMETERS_SMOOTHNESS = 0.005 # model parameters smoothness
LOSS_MOVING_AVERAGE = 0.9
