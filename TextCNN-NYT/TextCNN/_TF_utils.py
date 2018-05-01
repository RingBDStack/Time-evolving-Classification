

import tensorflow as tf
import _option as OPTION
import os
import numpy as np
import math



def activation_summary(x):
    """ Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)  # Outputs a Summary protocol buffer with a histogram.
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))  # Outputs a Summary protocol buffer with scalar values.

def variable_with_weight_decay(name, initial_value, dtype = tf.float32, trainable = True, wd=None):
    """ Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        initial_value: initial value for Variable
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    var = tf.Variable(initial_value=initial_value, name=name, trainable=trainable, dtype=dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name + '_loss')
        tf.add_to_collection('losses', weight_decay)
    return var






def smoothness_decay_func(t, i, type = 0, speed = math.e, factor = 1):
    if type ==0:
        # exponential decay
        # a*b^(t-i)
        # b=1/2, 1/e, 1/10
        return factor * math.pow(speed, -(t - i))
    elif type == 1:
        return 1.0
    elif type ==2:
        return 1.0 if i==t-1 else 0.0

def calculate_tensors_distance(tensor_v1,tensor_v2, type='euclidean'):
    # compute v1 and v2's distance

    if type == 'euclidean':
        # Euclidean distance
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tensor_v1, tensor_v2))))
    elif type == 'cosin':
        # cosin distance
        distance = tf.divide(tf.reduce_sum(tf.multiply(tensor_v1, tensor_v2)),
                             tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(tensor_v1))),
                                         tf.sqrt(tf.reduce_sum(tf.square(tensor_v2)))))

    return distance

def calculate_para_dependence_loss_t(model_para,t):
    # Calculate the Variable's dependency constraint
    pre_model_para = []
    lineslist = open(os.path.join(OPTION.MODELPARA_DIR, 'para_%d_best' % t), 'r').readlines()
    for line in lineslist:
        line = line.strip()
        linelist = line.split(' ')
        linelist = [float(i) for i in linelist]
        vector = np.array(linelist,dtype = np.float32)
        vector = tf.constant(vector,dtype=tf.float32)
        # vector = tf.reshape(vector, [-1])
        pre_model_para.append(vector)

    distance_list = []
    for j in range(0, len(model_para)):
        distance = calculate_tensors_distance(model_para[j], pre_model_para[j], type='euclidean')
        distance_list.append(distance)
    return tf.add_n(distance_list)

def calculate_para_dependence_loss(model_para,t):
    # t is current timestep, model_para is current model paramaters
    for i in range(0,t):
        distance = calculate_para_dependence_loss_t(model_para,i)
        smoothness_decay = smoothness_decay_func(t-1,i,type = 0, factor = OPTION.PARAMETERS_SMOOTHNESS)
        dependencies = tf.multiply(distance, smoothness_decay,
                                   name='smoothness_loss_para_%d' % i)
        tf.add_to_collection('losses', dependencies)





def calculate_cross_entropy_loss(logits, labels, UseWeight = False, Weight = None):
    """
    Calculate Mean cross-entropy loss
    Args:
        logits: Logits from inference(), 2D tensor of [None, NUM_CLASSES].
        labels: 2D tensor of [None, NUM_CLASSES].
    Returns:
    """
    with tf.name_scope('loss'):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                     name='loss_cross_entropy')
        if UseWeight:
            loss_cross_entropy_mean = tf.reduce_mean(tf.multiply(
                loss_cross_entropy,
                Weight), name = 'loss_cross_entropy_mean')
        else:
            loss_cross_entropy_mean = tf.reduce_mean(loss_cross_entropy,
                                                     name = 'loss_cross_entropy_mean')

        tf.add_to_collection('losses', loss_cross_entropy_mean)
        return loss_cross_entropy

def calculate_cross_entropy_loss_with_MovingAverage(logits, labels, last_loss):
    """
    Calculate Mean cross-entropy loss
    Args:
        logits: Logits from inference(), 2D tensor of [None, NUM_CLASSES].
        labels: 2D tensor of [None, NUM_CLASSES].
    Returns:
    """
    with tf.name_scope('loss'):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        last_loss = tf.cast(last_loss, tf.float32)
        loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                     name='loss_cross_entropy')
        loss_cross_entropy_moving_average = tf.add(
            tf.multiply(loss_cross_entropy, OPTION.LOSS_MOVING_AVERAGE),
            tf.multiply(last_loss, 1.0 - OPTION.LOSS_MOVING_AVERAGE),
            name='loss_cross_entropy_moving_average')
        loss_cross_entropy_mean = tf.reduce_mean(loss_cross_entropy_moving_average,
                                                 name = 'loss_cross_entropy_mean')

        tf.add_to_collection('losses', loss_cross_entropy_mean)

        return loss_cross_entropy_moving_average

def calculate_accuracy(logits, labels):
    # accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def add_loss_summaries(total_loss):
    """ Add summaries for losses.
        Generates moving average for all losses and associated summaries for visualizing the performance of the network.
        moving average -> eliminate noise

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    # The moving averages are computed using exponential decay:
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)   equivalent to:
    # shadow_variable = decay * shadow_variable + (1 - decay) * variable
    loss_averages = tf.train.ExponentialMovingAverage(OPTION.MOVING_AVERAGE_DECAY, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """ Create an optimizer and apply to all trainable variables.
            Add moving average for all trainable variables.

        Args:
            total_loss: total loss from loss().
            global_step: Integer Variable counting the number of training steps processed.

        Returns:
            train_op: op for training.
    """
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.MomentumOptimizer(lr, graphcnn_option.MOMENTUM)
        opt = tf.train.AdamOptimizer(OPTION.INITIAL_LEARNING_RATE)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        OPTION.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op




def calculate_loss_weight(logits,labels,last_logits):
    labels = tf.cast(labels, tf.float32)
    a = tf.nn.relu(tf.reduce_sum(tf.multiply(tf.subtract(last_logits,logits),labels),1))
    b = tf.subtract(1.0, a)
    out = tf.square(b)
    return out

