

## path
DATA_PATH = '../data' # Path to data directory
TRAIN_DIR = './tmp/train'  # Directory where to write event logs and checkpoint.
EVAL_DIR = './tmp/eval'
MODELPARA_DIR = './tmp'
CHECKPOINT_DIR = './tmp/train'
PRE_TRAIN_MODEL = './tmp/pre_train_model'

## file
TRAIN_DATA_NAME = 'train_data'
TEST_DATA_NAME = 'test_data'
DATA_OPTION_NAME = 'option'
DATA_VEC_NAME = 'word2vec_RCV1.wv'




## model parameters
DROPOUT_KEEP_PROB = 0.5  # Add a dropout during training.
INITIAL_LEARNING_RATE = 0.001 # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.999 # The decay to use for the moving average.
WEIGHT_DECAY = 0.0005     # 0.00005  # 0.0005 # l2 regularization weight decay


BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
EMEBEDDING_DIMENSION = 100
MAX_CKPT_PERIOD = 250 #1000 #3000 # Dimensionality of character embedding
NUM_EPOCHS = 1000  # Number of epochs to run.
MIN_CKPTS = 3
SUMMARY_PERIOD = 30

NUM_CLASSES = 12
SEQUENCE_LEN = 128



PARAMETERS_SMOOTHNESS = 0.0005 # model parameters smoothness
LOSS_MOVING_AVERAGE = 0.9