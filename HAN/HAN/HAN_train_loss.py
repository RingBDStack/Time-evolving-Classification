


from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import math
import shutil

import _option as OPTION
import _TF_utils as myTF
import HAN_model as MODEL
import HAN_input as INPUT



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_integer('eval_interval_secs', 60, """How often to run the eval.""")
tf.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


os.environ['CUDA_VISIBLE_DEVICES']='1'


def evaluate(trainDataSet,time,model_time):

    with tf.Graph().as_default() as g, tf.device('/cpu:0'):

        # Placeholders for input, output and dropout
        input_x = tf.placeholder(tf.int32, [None, OPTION.SEQUENCE_LEN, OPTION.SENT_LEN], name="input_x")
        input_y = tf.placeholder(tf.int32, [None, OPTION.NUM_CLASSES], name="input_y")

        han = MODEL.Model(sequence_length = OPTION.SEQUENCE_LEN, sent_length = OPTION.SENT_LEN,
                              num_classes=OPTION.NUM_CLASSES,
                              vocab_size=None,
                              embedding_size=OPTION.EMEBEDDING_DIMENSION,
                              Word2vec=True, Trainable=False)

        # inference model.
        logits, _ = han.inference(input_x, eval_data=True)

        # Calculate loss.
        loss = myTF.calculate_cross_entropy_loss(logits, input_y)

        # get model paramaters
        # paramaters_list_reshape = han.get_paramaters_list_reshape()

        # Restore the moving average version of the learned variables for eval. # ?????????????????????????
        variable_averages = tf.train.ExponentialMovingAverage(OPTION.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(OPTION.EVAL_DIR, g)


        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)) as sess:

            if os.path.exists(os.path.join(OPTION.EVAL_DIR, 'model.ckpt-best.index')):
                # new_saver = tf.train.import_meta_graph(
                #     os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-' + checkpoint + '.meta'))
                saver.restore(sess,
                              os.path.join(OPTION.EVAL_DIR, 'model.ckpt-best'))
            else:
                print('No checkpoint file found')
                return

            max_steps_per_epoch = int(math.ceil( trainDataSet.get_dataset_size() / float(OPTION.EVAL_BATCH_SIZE)))
            total_predicted_value = []
            for step in range(max_steps_per_epoch):
                train_data, train_label = trainDataSet.next_batch(OPTION.EVAL_BATCH_SIZE)
                predicted_value = sess.run(loss,
                                           feed_dict={input_x: train_data, input_y: train_label})
                total_predicted_value.append(predicted_value)

            # test_data, test_label = evalDataSet.next_batch(OPTION.EVAL_BATCH_SIZE)
            summary = tf.Summary()
            # summary.ParseFromString(sess.run(summary_op, feed_dict={input_x: test_data, input_y: test_label}))

            total_predicted_value = np.concatenate(total_predicted_value,axis=0)

            assert trainDataSet.get_dataset_size() == total_predicted_value.shape[0], 'sample_count error!'

            detail_filename = os.path.join(OPTION.MODELPARA_DIR, 'loss_%d_%d' %(time,model_time))
            if os.path.exists(detail_filename):
                os.remove(detail_filename)
            np.savetxt(detail_filename, total_predicted_value, fmt='%f')




def main(argv=None):
    time_list = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, 'time_list'), dtype=int)
    time_list = np.reshape(time_list, [-1])
    model_time = time_list[-1]
    ans = input('please choose times to get train loss(next time):')
    time_list = np.array((ans.strip()).split(' '), dtype=int)
    print('...')
    for one in time_list:
        evalDataSet = INPUT.generate_train_data([one],shuffled=False,label_used=True)
        evaluate(evalDataSet,one,model_time)


if __name__ == '__main__':
    tf.app.run()








