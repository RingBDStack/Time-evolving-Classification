


from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import math
import shutil

import _option as OPTION
import RCNNpn_model as EC_MODEL
import RCNNpn_input as EC_INPUT



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_integer('eval_interval_secs', 60, """How often to run the eval.""")
tf.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


os.environ['CUDA_VISIBLE_DEVICES']='0'


def evaluate(evalDataSet,time,model_time,name='train'):

    with tf.Graph().as_default() as g, tf.device('/gpu:0'):

        # Placeholders for input, output and dropout
        input_x = tf.placeholder(tf.int32, [None, OPTION.SEQUENCE_LEN], name="input_x")
        input_left = tf.placeholder(tf.int32, [None, OPTION.SEQUENCE_LEN], name="input_left")
        input_right = tf.placeholder(tf.int32, [None, OPTION.SEQUENCE_LEN], name="input_right")
        batch_size = tf.placeholder(tf.int32, [],name="batch_size")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        feature_size = OPTION.FEATURE_SIZE * min(model_time - 0, OPTION.DP_DEPTH)
        if feature_size > 0:
            features_before = tf.placeholder(tf.float32, [None, feature_size], name="features_before")
        else:
            features_before = None

        textcnn = EC_MODEL.Model(sequence_length=OPTION.SEQUENCE_LEN,
                                 num_classes=OPTION.NUM_CLASSES,
                                 vocab_size=None,
                                 embedding_size=OPTION.EMEBEDDING_DIMENSION,
                                 context_size=OPTION.CONTEXT_DIMENSION,
                                 feature_size=OPTION.FEATURE_SIZE,
                                 Word2vec=True, Trainable=False)

        # inference model.
        _, features = textcnn.inference(input_x,input_left,input_right, features_before, batch_size, keep_prob)

        # get model paramaters
        # paramaters_list_reshape = textcnn.get_paramaters_list_reshape()

        # Restore the moving average version of the learned variables for eval. # ?????????????????????????
        variable_averages = tf.train.ExponentialMovingAverage(OPTION.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(OPTION.EVAL_DIR, g)
        
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
        # config.gpu_options.allow_growth = OPTION.MEMORY_ALLOW_GROWTH  # 程序按需申请内存

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        with tf.Session(config=config) as sess:

            if os.path.exists(os.path.join(OPTION.EVAL_DIR, 'model.ckpt-best.index')):
                # new_saver = tf.train.import_meta_graph(
                #     os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-' + checkpoint + '.meta'))
                saver.restore(sess,
                              os.path.join(OPTION.EVAL_DIR, 'model.ckpt-best'))
            else:
                print('No checkpoint file found')
                return

            max_steps_per_epoch = int(math.ceil( evalDataSet.get_dataset_size() / float(OPTION.EVAL_BATCH_SIZE)))
            total_predicted_value = []
            for step in range(max_steps_per_epoch):
                test_data, test_left, test_right, test_features= evalDataSet.next_batch(OPTION.EVAL_BATCH_SIZE)
                if feature_size > 0:
                    feed_dict = {input_x: test_data, batch_size: np.size(test_data, 0), keep_prob: 1.0,
                                 input_left: test_left, input_right: test_right,
                                 features_before: test_features}
                else:
                    feed_dict = {input_x: test_data, batch_size: np.size(test_data, 0), keep_prob: 1.0,
                                 input_left: test_left, input_right: test_right}
                predicted_value = sess.run(features,
                                           feed_dict=feed_dict)
                total_predicted_value.append(predicted_value)

            # test_data, test_label = evalDataSet.next_batch(OPTION.EVAL_BATCH_SIZE)
            summary = tf.Summary()
            # summary.ParseFromString(sess.run(summary_op, feed_dict={input_x: test_data, input_y: test_label}))

            total_predicted_value = np.concatenate(total_predicted_value,axis=0)

            total_predicted_value = total_predicted_value[0:evalDataSet.get_dataset_size()]

            assert evalDataSet.get_dataset_size() == total_predicted_value.shape[0], 'sample_count error!'

            detail_filename = os.path.join(OPTION.MODELPARA_DIR, 'features_%s_%d_%d' % (name,time,model_time))
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_predicted_value, fmt='%.4f')




def main(argv=None):
    time_list = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, 'time_list'), dtype=int)
    time_list = np.reshape(time_list, [-1])
    model_time = time_list[-1]
    ans = input('please choose times to get features:')
    time_list = np.array((ans.strip()).split(' '), dtype=int)
    print('...')
    for one in time_list:
        evalDataSet = EC_INPUT.generate_feature_data(one,model_time,shuffled=False,label_used=False,
                                                            isTrain=True)
        evaluate(evalDataSet,one,model_time,name='train')
        evalDataSet = EC_INPUT.generate_feature_data(one,model_time, shuffled=False, label_used=False,
                                                            isTrain=False)
        evaluate(evalDataSet, one, model_time,name='test')


if __name__ == '__main__':
    tf.app.run()








