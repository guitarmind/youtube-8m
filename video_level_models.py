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
import model_utils

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


class MLPModel(models.BaseModel):
  """MLP model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a MLP model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = model_utils.make_fully_connected_net(model_input,
        [512, 256], vocab_size, l2_penalty)
    return {"predictions": output}

class MLPModelV2(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fully_connected_net(model_input,
            [784, 512, 256], vocab_size, l2_penalty)
        return {"predictions": output}

class MLPModel384(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fully_connected_net(model_input,
            [784, 512, 384], vocab_size, l2_penalty)
        return {"predictions": output}

class DeepMLPModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fully_connected_net(model_input,
            [784, 512, 512, 512, 256], vocab_size, l2_penalty)
        return {"predictions": output}

class BatchNormMLP(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fully_connected_net(model_input,
            [784, 512, 512, 512, 256], vocab_size, l2_penalty, batch_norm=True)
        return {"predictions": output}

class SkipConnections(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fcnet_with_skips(model_input,
            [784, 512, 512, 512, 256], [(0, 3), (2, 4)], vocab_size, l2_penalty)
        return {"predictions": output}

class DeepSkip(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fcnet_with_skips(model_input,
            [784] + [512]*8,
            [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
        return {"predictions": output}

class BigNN(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fcnet_with_skips(model_input,
            [1024] + [768]*8,
            [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
        return {"predictions": output}

class BiggerNN(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fcnet_with_skips(model_input,
            [1536] + [1024]*8,
            [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
        return {"predictions": output}

class DeeperSkip(models.BaseModel):
    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        output = model_utils.make_fcnet_with_skips(model_input,
            [784] + [512]*14,
            [(0, 3), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)],
            vocab_size, l2_penalty)
        return {"predictions": output}


## Video Level Models by umudev/youtube-8m ##


class TwoLayerModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_hidden_units=2048, l2_penalty=1e-8, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    hidden1 = slim.fully_connected(
        model_input, num_hidden_units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_1')

    hidden1 = slim.dropout(hidden1, 0.5, scope=prefix+"dropout1")

    output = slim.fully_connected(
        hidden1, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_2')

    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm,"hidden_features": hidden1}
    #return {"predictions": output}

class NeuralAverageModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048a/')
    output_2048b = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048b/')
    output_2048c = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048c/')
    output_2048d = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048d/')

    t1 = output_2048a["predictions"]
    t2 = output_2048b["predictions"]
    t3 = output_2048c["predictions"]
    t4 = output_2048d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])

    scalar = tf.constant(0.25)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class StackModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a Stack of Neural Networks Model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048a/')
    output_2048b = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048b/')
    output_2048c = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048c/')
    output_2048d = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048d/')

    t1 = output_2048a["hidden_features"]
    t2 = output_2048a["hidden_features"]
    t3 = output_2048a["hidden_features"]
    t4 = output_2048a["hidden_features"]

    stacked_features = tf.concat([t1, t2, t3, t4], 1)
    stacked_fc1 = slim.fully_connected(
      stacked_features,
      2048,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="Stack/fc1")
    stacked_fc1 = slim.dropout(stacked_fc1, 0.5, scope="Stack/dropout1")
    stacked_fc2 = slim.fully_connected(
      stacked_fc1,
      vocab_size,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="Stack/fc2")

    output = tf.nn.sigmoid(stacked_fc2)

    #return {"predictions": output, "regularization_loss": weights_norm}
    return {"predictions": output}


## Video Level Models by DeepVoltaire/youtube-8m ##


class FCBNModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, nb_units=2000, **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(model_input, nb_units, scope="fc1")
    output = slim.batch_norm(output, scope="bn1")
    output = slim.dropout(output, 0.5, scope="dropout1")
    output = slim.fully_connected(output, nb_units, scope="fc2")
    output = slim.batch_norm(output, scope="bn2")
    output = slim.dropout(output, 0.5, scope="dropout2")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc3")
    return {"predictions": output}
