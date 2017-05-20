# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


##  Video Level CNN Model by ehumss/youtube-8m ##

class CNNModel(models.BaseModel):
  """Conv model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(model_input, 64, [4, 4])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [4, 4])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        output = slim.fully_connected(net, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
        
    return {"predictions": output}


## Video Level Models by vivekn/youtube-8m ##

class AvgPoolLSTM(models.BaseModel):

    def create_model(self,
        model_input,
        vocab_size,
        num_frames,
        pool_window=20,
        l2_penalty=1e-8,
        **unused_params):
        input_4d = tf.transpose(tf.stack([model_input]), perm=[1, 2, 3, 0])
        pooled_input = tf.squeeze(tf.nn.avg_pool(input_4d,
            [1, pool_window, 1, 1], [1, pool_window, 1, 1], 'VALID'), axis=[3])
        num_frames_scaled = tf.to_int32(
            tf.ceil(tf.to_float(num_frames) / pool_window))

        lstm_size = FLAGS.lstm_cells
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, pooled_input,
            sequence_length=num_frames_scaled, dtype=tf.float32)

        output = utils.make_fully_connected_net(
            tf.transpose(outputs, perm=[1, 0, 2])[-1],
            [],
            vocab_size,
            l2_penalty
        )
        return {"predictions": output}

class MeanLSTM(models.BaseModel):

    def create_model(self,
        model_input,
        vocab_size,
        num_frames,
        pool_window=20,
        l2_penalty=1e-8,
        **unused_params):
        input_4d = tf.transpose(tf.stack([model_input]), perm=[1, 2, 3, 0])
        pooled_input = tf.squeeze(tf.nn.avg_pool(input_4d,
            [1, pool_window, 1, 1], [1, pool_window, 1, 1], 'VALID'), axis=[3])
        num_frames_scaled = tf.to_int32(
            tf.ceil(tf.to_float(num_frames) / pool_window))

        lstm_size = FLAGS.lstm_cells
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, pooled_input,
            sequence_length=num_frames_scaled, dtype=tf.float32)
        lstm_out = tf.transpose(outputs, perm=[1, 0, 2])[-1]
        mean_pooled = utils.get_avg_pooled(model_input, num_frames)

        output = utils.make_fully_connected_net(
            tf.concat([lstm_out, mean_pooled], 1),
            [512, 256],
            vocab_size,
            l2_penalty
        )
        return {"predictions": output}

class ConvolutionalLSTM(models.BaseModel):

    def create_model(self,
        model_input,
        vocab_size,
        num_frames,
        pool_window=20,
        l2_penalty=1e-8,
        **unused_params):
        input_4d = tf.transpose(tf.stack([model_input]), perm=[1, 2, 0, 3])

        conv1 = utils.make_conv_relu_pool(input_4d, 3, 4, 256)
        conv2 = utils.make_conv_relu_pool(conv1, 3, 4, 256)

        lstm_size = FLAGS.lstm_cells
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, tf.squeeze(conv2, 2),
            dtype=tf.float32)
        lstm_out = tf.transpose(outputs, perm=[1, 0, 2])[-1]
        mean_pooled = utils.get_avg_pooled(model_input, num_frames)

        output = utils.make_fully_connected_net(
            tf.concat([lstm_out, mean_pooled], 1),
            [784, 512, 256],
            vocab_size,
            l2_penalty
        )
        return {"predictions": output}


class ConvolutionalLSTMSmall(models.BaseModel):

    def create_model(self,
        model_input,
        vocab_size,
        num_frames,
        pool_window=20,
        l2_penalty=1e-8,
        **unused_params):
        input_4d = tf.transpose(tf.stack([model_input]), perm=[1, 2, 0, 3])

        conv1 = utils.make_conv_relu_pool(input_4d, 3, 4, 64, batch_norm=True)
        conv2 = utils.make_conv_relu_pool(conv1, 3, 4, 64, batch_norm=True)
        conv3 = utils.make_conv_relu_pool(conv2, 3, 4, 64, batch_norm=True)

        lstm_size = FLAGS.lstm_cells
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0,
            use_peepholes=True)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, tf.squeeze(conv3, 2),
            dtype=tf.float32)
        lstm_out = tf.transpose(outputs, perm=[1, 0, 2])[-1]
        mean_pooled = utils.get_avg_pooled(model_input, num_frames)

        output = utils.make_fcnet_with_skips(
            tf.concat([lstm_out, mean_pooled], 1),
            [784, 512, 512, 512, 256], [(0, 3), (2, 4)], vocab_size, l2_penalty)
        return {"predictions": output}

class FramePooler(models.BaseModel):
    """
    Simple 9 layer fully connected model with inputs as
    (raw_features, max_pooled, min_pooled, avg_pooled, l2_norm)
    """
    def create_model(self,
        model_input,
        vocab_size,
        num_frames,
        l2_penalty=1e-8,
        **unused_params):
        avg_pooled = utils.get_avg_pooled(model_input, num_frames)
        features = tf.concat([
            avg_pooled,
            utils.get_standard_dev_and_l2(model_input, num_frames, avg_pooled),
            utils.get_min_max_pooled(model_input, num_frames),
            tf.expand_dims(tf.to_float(num_frames), 1),
        ], 1)
        output = utils.make_fcnet_with_skips(features, [1024]+[768]*8,
            [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
        return {"predictions": output}


