from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import math
import shutil

import _option as OPTION
import HAN_model
import HAN_input
import _TF_utils as myTF

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_integer('eval_interval_secs', 0.5, """How often to run the eval.""")
tf.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


os.environ['CUDA_VISIBLE_DEVICES']='3'

# def calc_eval_loss(total_predicted_value, total_true_value):
#     with tf.Graph().as_default():
#         total_logits = tf.placeholder(tf.float32, [None, OPTION.NUM_CLASSES], name="total_logits")
#         total_labels = tf.placeholder(tf.float32, [None, OPTION.NUM_CLASSES], name="total_labels")
#         total_loss = HAN_model.calculate_loss(total_logits, total_labels)
#         with tf.Session(config=tf.ConfigProto(
#                 allow_soft_placement=FLAGS.allow_soft_placement,
#                 log_device_placement=FLAGS.log_device_placement)) as sess:
#             return total_loss.eval(feed_dict={total_logits: total_predicted_value,
#                                                    total_labels: total_true_value},
#                                    session=sess)


def evaluation_result(data_path, total_predicted_value, total_true_value, total_loss, global_step, best_eval_value,
                      summary):
    detail_filename = os.path.join(data_path, 'log_eval_for_predicted_value_dictribution')
    if os.path.exists(detail_filename):
        os.remove(detail_filename)
    np.savetxt(detail_filename, total_predicted_value, fmt='%.4f')

    detail_filename = os.path.join(data_path, 'log_eval_for_cross_entropy_loss')
    if os.path.exists(detail_filename):
        os.remove(detail_filename)
    np.savetxt(detail_filename, total_loss, fmt='%.4f')

    total_prediction = np.argmax(total_predicted_value, 1)
    total_truth = np.argmax(total_true_value, 1)
    total_prediction_and_truth = np.concatenate((np.reshape(total_prediction, [-1, 1]),
                                                 np.reshape(total_truth, [-1, 1])), axis=1)
    accuracy = float(np.sum(total_prediction == total_truth)) / np.size(total_prediction, 0)
    summary.value.add(tag='accuracy', simple_value=accuracy)

    detail_filename = os.path.join(data_path, 'log_eval_for_prediction_and_truth')
    if os.path.exists(detail_filename):
        os.remove(detail_filename)
    np.savetxt(detail_filename, total_prediction_and_truth, fmt='%d')

    filename_eval_log = os.path.join(data_path, 'log_eval')
    file_eval_log = open(filename_eval_log, 'a')
    np.set_printoptions(threshold=np.nan)

    loss = np.sum(total_loss) / len(total_loss)

    print('%s, ckpt-%d: acc=%.4f, loss=%.4f, num=%d' % (datetime.now(), global_step, accuracy, loss,
                                                        np.size(total_predicted_value, axis=0)), file=file_eval_log)
    print('%s, ckpt-%d: acc=%.4f, loss=%.4f, num=%d' % (datetime.now(), global_step, accuracy, loss,
                                                        np.size(total_predicted_value, axis=0)))

    for label_i in range(0, OPTION.NUM_CLASSES):
        prediction = total_prediction == label_i
        expectation = total_truth == label_i
        equal = prediction & expectation
        preNum = sum(prediction)
        expecNum = sum(expectation)
        equal_num = sum(equal)
        precise = equal_num / float(preNum) if preNum != 0 else 0
        recall = equal_num / float(expecNum) if expecNum != 0 else 0
        F1 = ((precise * recall) * 2) / (precise + recall) if (precise + recall) != 0 else 0
        print('    label %d: precise=%.4f (%d/%d), recall=%.4f (%d/%d), F1-Measure=%.4f' %
              (label_i, precise, equal_num, preNum, recall, equal_num, expecNum, F1), file=file_eval_log)
        summary.value.add(tag='label%d_precise' % label_i, simple_value=precise)
        summary.value.add(tag='label%d_recall' % label_i, simple_value=recall)
        summary.value.add(tag='label%d_F1' % label_i, simple_value=F1)
    file_eval_log.close()

    if accuracy > best_eval_value:
        best_eval_value = accuracy
        filename_eval_best = os.path.join(data_path, 'best_eval')
        file_eval_best = open(filename_eval_best, 'w')
        print('best eval: ckpt-%d, acc=%.4f, loss=%.4f' % (global_step, accuracy, loss), file=file_eval_best)
        file_eval_best.close()
        sourceFile = os.path.join(data_path, 'log_eval_for_predicted_value_dictribution')
        targetFile = os.path.join(data_path, 'best_eval_for_predicted_value_dictribution')
        if os.path.exists(targetFile):
            os.remove(targetFile)
        shutil.copy(sourceFile, targetFile)
        sourceFile = os.path.join(data_path, 'log_eval_for_prediction_and_truth')
        targetFile = os.path.join(data_path, 'best_eval_for_prediction_and_truth')
        if os.path.exists(targetFile):
            os.remove(targetFile)
        shutil.copy(sourceFile, targetFile)
        sourceFile = os.path.join(data_path, 'log_eval_for_cross_entropy_loss')
        targetFile = os.path.join(data_path, 'best_eval_for_cross_entropy_loss')
        if os.path.exists(targetFile):
            os.remove(targetFile)
        shutil.copy(sourceFile, targetFile)

        sourceFile = os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-%d.data-00000-of-00001' % global_step)
        targetFile = os.path.join(data_path, 'model.ckpt-best.data-00000-of-00001')
        if os.path.exists(targetFile):
            os.remove(targetFile)
        shutil.copy(sourceFile, targetFile)
        sourceFile = os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-%d.index' % global_step)
        targetFile = os.path.join(data_path, 'model.ckpt-best.index')
        if os.path.exists(targetFile):
            os.remove(targetFile)
        shutil.copy(sourceFile, targetFile)
        sourceFile = os.path.join(OPTION.TRAIN_DIR, 'model.ckpt-%d.meta' % global_step)
        targetFile = os.path.join(data_path, 'model.ckpt-best.meta')
        if os.path.exists(targetFile):
            os.remove(targetFile)
        shutil.copy(sourceFile, targetFile)

    return best_eval_value


def evaluate(evalDataSet, ckpt_value, eval_value):
    with tf.Graph().as_default() as g, tf.device('/gpu:0'):

        # Placeholders for input, output and dropout
        input_x = tf.placeholder(tf.int32, [None, OPTION.SEQUENCE_LEN, OPTION.SENT_LEN], name="input_x")
        input_y = tf.placeholder(tf.int32, [None, OPTION.NUM_CLASSES], name="input_y")

        han = HAN_model.Model(sequence_length=OPTION.SEQUENCE_LEN, sent_length=OPTION.SENT_LEN,
                              num_classes=OPTION.NUM_CLASSES,
                              vocab_size=None,
                              embedding_size=OPTION.EMEBEDDING_DIMENSION,
                              Word2vec=True, Trainable=False)

        # inference model.
        logits, _ = han.inference(input_x, eval_data=True)

        # Calculate loss.
        loss = myTF.calculate_cross_entropy_loss(logits, input_y)

        logits = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval. # ?????????????????????????
        variable_averages = tf.train.ExponentialMovingAverage(OPTION.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(OPTION.EVAL_DIR, g)

        last_eval_ckpt = ckpt_value
        best_eval_value = eval_value

        config=tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存

        while True:
            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU implementations.
            with tf.Session(config=config) as sess:
                ckpt = tf.train.get_checkpoint_state(OPTION.CHECKPOINT_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    # extract global_step
                    global_step_for_restore = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    if global_step_for_restore > last_eval_ckpt:
                        # Restores from checkpoint
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    else:
                        if tf.gfile.Exists("TRAIN_SUCCEED"):
                            print("Train terminated, eval terminating...")
                            return
                else:
                    print('No checkpoint file found')
                    time.sleep(FLAGS.eval_interval_secs)
                    continue

                if global_step_for_restore > last_eval_ckpt:
                    max_steps_per_epoch = int(math.ceil(evalDataSet.get_dataset_size() / float(OPTION.EVAL_BATCH_SIZE)))
                    start_time = time.time()
                    total_predicted_value = []
                    total_true_value = []
                    total_loss = []
                    for step in range(max_steps_per_epoch):
                        test_data, test_label = evalDataSet.next_batch(OPTION.EVAL_BATCH_SIZE)
                        predicted_value, true_value, loss_value = sess.run([logits, input_y, loss],
                                                                           feed_dict={input_x: test_data,
                                                                                      input_y: test_label})
                        total_predicted_value.append(predicted_value)
                        total_true_value.append(true_value)
                        total_loss.append(loss_value)
                    duration = time.time() - start_time

                    # test_data, test_label = evalDataSet.next_batch(OPTION.EVAL_BATCH_SIZE)
                    summary = tf.Summary()
                    # summary.ParseFromString(sess.run(summary_op, feed_dict={input_x: test_data, input_y: test_label}))

                    total_predicted_value = np.concatenate(total_predicted_value, axis=0)
                    total_true_value = np.concatenate(total_true_value, axis=0)
                    total_loss = np.concatenate(total_loss, axis=0)

                    assert evalDataSet.get_dataset_size() == total_predicted_value.shape[0], 'sample_count error!'
                    best_eval_value = evaluation_result(OPTION.EVAL_DIR, total_predicted_value, total_true_value,
                                                        total_loss,
                                                        global_step_for_restore, best_eval_value, summary)
                    summary_writer.add_summary(summary, global_step_for_restore)

                    last_eval_ckpt = global_step_for_restore

            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    eval_value = 0
    ckpt_value = 0
    if tf.gfile.Exists(OPTION.EVAL_DIR):
        print('the evaluate data has already exists!')
        str = input('whether to delete the old evaluate directory:(y/n)')
        if str == 'y' or str == 'Y':
            tf.gfile.DeleteRecursively(OPTION.EVAL_DIR)
        elif str == 'n' or str == 'N':
            print('restoring...')
            line = open(os.path.join(OPTION.EVAL_DIR, 'best_eval'), 'r').readline()
            eval_value = float(line.strip().split(',')[1].split('=')[1])
            line = open(os.path.join(OPTION.EVAL_DIR, 'log_eval'), 'r').readlines()[-5]
            ckpt_value = int(line.strip().split(',')[1].split(':')[0].split('-')[1])
        else:
            print('invalid input!')
            return
    tf.gfile.MakeDirs(OPTION.EVAL_DIR)

    # checkpoint = input('please input the choosed checkpoint to eval:(0 for latest)')
    ans = input('please choose times to eval:')
    time_list = np.array((ans.strip()).split(' '), dtype=int)

    evalDataSet = HAN_input.generate_eval_data(time_list)

    print('evaluating...')

    evaluate(evalDataSet, ckpt_value, eval_value)
    open("EVAL_SUCCEED", "w")


if __name__ == '__main__':
    tf.app.run()
