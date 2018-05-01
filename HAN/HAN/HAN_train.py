from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import math

import _option as OPTION
import _TF_utils as myTF
import HAN_model
import HAN_input

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


os.environ['CUDA_VISIBLE_DEVICES']='0'



def train(newTrain, checkpoint, trainDataSet):
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Placeholders for input, output and dropout
        input_x = tf.placeholder(tf.int32, [None, OPTION.SEQUENCE_LEN, OPTION.SENT_LEN], name="input_x")
        input_y = tf.placeholder(tf.int32, [None, OPTION.NUM_CLASSES], name="input_y")

        han = HAN_model.Model(sequence_length=OPTION.SEQUENCE_LEN, sent_length=OPTION.SENT_LEN,
                              num_classes=OPTION.NUM_CLASSES,
                              vocab_size=None,
                              embedding_size=OPTION.EMEBEDDING_DIMENSION,
                              Word2vec=True, Trainable=False)

        # inference model.
        logits, _ = han.inference(input_x)

        # Calculate loss.
        loss = HAN_model.calculate_loss(logits, input_y)

        # Calculate accuracy
        accuracy = myTF.calculate_accuracy(logits, input_y)

        # updates the model parameters.
        train_op = myTF.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        config=tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
        config.gpu_options.allow_growth = OPTION.MEMORY_ALLOW_GROWTH  # 程序按需申请内存
        sess = tf.Session(config=config)

        first_step = 0
        if not newTrain:
            print('restoring...')
            if checkpoint == '0':  # choose the latest one
                ckpt = tf.train.get_checkpoint_state(OPTION.TRAIN_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    # new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step_for_restore = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    first_step = int(global_step_for_restore) + 1
                else:
                    print('No checkpoint file found')
                    return
            else:  #
                if os.path.exists(os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-' + checkpoint + '.index')):
                    # new_saver = tf.train.import_meta_graph(
                    #     os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-' + checkpoint + '.meta'))
                    saver.restore(sess,
                        os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-' + checkpoint))
                    first_step = int(checkpoint) + 1
                else:
                    print('No checkpoint file found')
                    return
        else:
            sess.run(init)
            if os.path.exists(os.path.join(OPTION.PRE_TRAIN_MODEL, 'model.ckpt-pretrain.index')):
                # saver_load = tf.train.Saver(var_list=tf.get_collection('pretrained_variables'))
                saver_load = tf.train.Saver(var_list=tf.trainable_variables())
                print('load pretrained variables...')
                saver_load.restore(sess, os.path.join(OPTION.PRE_TRAIN_MODEL, 'model.ckpt-pretrain'))

        summary_writer = tf.summary.FileWriter(OPTION.TRAIN_DIR, sess.graph)

        filename_train_log = os.path.join(OPTION.TRAIN_DIR, 'log_train')
        if os.path.exists(filename_train_log):
            file_train_log = open(filename_train_log, 'a')
        else:
            file_train_log = open(filename_train_log, 'w')

        max_steps_per_epoch = int(math.ceil(trainDataSet.get_dataset_size() / float(OPTION.BATCH_SIZE)))
        max_steps = max_steps_per_epoch * OPTION.NUM_EPOCHS

        # ckpt_period = max_steps_per_epoch // OPTION.MIN_CKPTS
        # if ckpt_period > OPTION.MAX_CKPT_PERIOD:
        #     ckpt_period = OPTION.MAX_CKPT_PERIOD
        ckpt_period = OPTION.MAX_CKPT_PERIOD
        for step in range(first_step, max_steps):
            train_data, train_label = trainDataSet.next_batch(OPTION.BATCH_SIZE)
            start_time = time.time()
            _, loss_value, accuracy_value, current_global_step = sess.run([train_op, loss, accuracy, global_step],
                                                                          feed_dict={input_x: train_data,
                                                                                     input_y: train_label})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            assert step + 1 == current_global_step, 'step:%d, current_global_step:%d' % (step, current_global_step)

            current_epoch = int(current_global_step / float(max_steps_per_epoch)) + 1
            current_step = current_global_step % max_steps_per_epoch

            if current_global_step % 10 == 0:
                sec_per_batch = float(duration)
                format_str = '%s: step=%d(%d/%d), loss=%.4f, acc=%.4f; %.3f sec/batch)' % (datetime.now(),
                                                                                           current_global_step,
                                                                                           current_step, current_epoch,
                                                                                           loss_value, accuracy_value,
                                                                                           sec_per_batch)
                print(format_str, file=file_train_log)
                print(format_str)

            if current_global_step % OPTION.SUMMARY_PERIOD == 0:
                summary_str = sess.run(summary_op,
                                       feed_dict={input_x: train_data, input_y: train_label})
                summary_writer.add_summary(summary_str, current_global_step)

            # Save the model checkpoint periodically. (named 'model.ckpt-global_step.meta')
            if current_global_step % ckpt_period == 0 or (current_global_step + 1) == max_steps:
                checkpoint_path = os.path.join(OPTION.TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=current_global_step)
        file_train_log.close()


def main(argv=None):
    newTrain = True
    checkpoint = 0
    # assert not tf.gfile.Exists(FLAGS.train_dir), 'please move the old train directory to pre_versions!'
    if tf.gfile.Exists(OPTION.TRAIN_DIR):
        ans = input('whether to open up a new training:(y/n)')
        if ans == 'y' or ans == 'Y':
            newTrain = True
            tf.gfile.DeleteRecursively(OPTION.TRAIN_DIR)
        elif ans == 'n' or ans == 'N':
            newTrain = False
            checkpoint = input('please input the choosed checkpoint to restore:(0 for latest)')
            time_list = np.loadtxt(os.path.join(OPTION.MODELPARA_DIR, 'time_list'), dtype=int)
            time_list = np.reshape(time_list, [-1])
        else:
            print('invalid input!')
            return
    if newTrain:
        tf.gfile.MakeDirs(OPTION.TRAIN_DIR)
        ans = input('please choose times to train:')
        time_list = np.array((ans.strip()).split(' '), dtype=int)
        ans = input('whether accumulate historical data to train:(y/n)')
        if ans == 'y' or ans == 'Y':
            time_list = np.array([i for i in range(time_list[0])] + time_list.tolist())
        elif ans == 'n' or ans == 'N':
            pass
        else:
            print('invalid input!')
            return
        np.savetxt(os.path.join(OPTION.MODELPARA_DIR, 'time_list'), time_list, fmt='%d')

    # update paras
    trainDataSet = HAN_input.generate_train_data(time_list)

    print('training...')

    train(newTrain, checkpoint, trainDataSet)
    open("TRAIN_SUCCEED", "w")


if __name__ == '__main__':
    tf.app.run()
