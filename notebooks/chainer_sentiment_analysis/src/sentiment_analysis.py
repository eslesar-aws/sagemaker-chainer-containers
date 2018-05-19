# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os
import json

import numpy as np
from six import BytesIO
import chainer
from chainer import training
from chainer import serializers
from chainer.training import extensions

import nets
from nlp_utils import convert_seq, split_text, normalize_text, transform_to_array

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(hyperparameters, num_gpus, output_data_dir, channel_input_dirs, model_dir):
    """
    This function is called by the Chainer container during training when running on SageMaker with
    values populated by the training environment.

    Args:
        hyperparameters (dict): map of hyperparameters given to the training job.
        num_gpus (int): number of gpus available to the container, determined by instance type.
        output_data_dir (str): path to the directory to write output artifacts to
        channel_input_dirs (dict): Dictionary mapping input channel names to local filesystem paths
        model_dir (str): path to the directory to write model artifacts to.

    Returns:
        a trained Chainer model

    For more on `train`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    train_data = np.load(os.path.join(channel_input_dirs['train'], 'train.npz'))['data']
    train_labels = np.load(os.path.join(channel_input_dirs['train'], 'train.npz'))['labels']

    test_data = np.load(os.path.join(channel_input_dirs['test'], 'test.npz'))['data']
    test_labels = np.load(os.path.join(channel_input_dirs['test'], 'test.npz'))['labels']

    vocab = np.load(os.path.join(channel_input_dirs['vocab'], 'vocab.npy')).tolist()

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    num_classes = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(num_classes))

    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 30)
    dropout = hyperparameters.get('dropout', 0.4)
    num_layers = hyperparameters.get('num_layers', 1)
    num_units = hyperparameters.get('num_units', 300)
    model_type = hyperparameters.get('model', 'cnn')
    num_loaders = hyperparameters.get('num_loaders', 1)

    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epochs))
    print('# Dropout: {}'.format(dropout))
    print('# Layers: {}'.format(num_layers))
    print('# Units: {}'.format(num_units))

    # Setup a model
    if model_type == 'rnn':
        Encoder = nets.RNNEncoder
    elif model_type == 'cnn':
        Encoder = nets.CNNEncoder
    elif model_type == 'bow':
        Encoder = nets.BOWMLPEncoder
    else:
        raise ValueError('model_type must be "rnn", "cnn", or "bow"')

    encoder = Encoder(n_layers=num_layers, n_vocab = len(vocab), n_units=num_units, dropout=dropout)
    model = nets.TextClassifier(encoder, num_classes)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if num_gpus > 1:
        devices = range(num_gpus)
        train_iters = [chainer.iterators.SerialIterator(i, batch_size) \
                    for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
        test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
        updater = training.updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                                converter=convert_seq, devices=range(num_gpus))
    else:
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
        test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
        updater = training.updater.StandardUpdater(train_iter, optimizer, converter=convert_seq, device=device)

    trainer = training.Trainer(updater, (epochs, 'epoch'), out=output_data_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, converter=convert_seq, device=device))

    # Take a best snapshot.
    record_trigger = training.triggers.MaxValueTrigger('validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Save additional model settings, which will be used to reconstruct the model during hosting
    model_setup = {}
    model_setup['num_classes'] = num_classes
    model_setup['model_type'] = model_type
    model_setup['num_layers'] = num_layers
    model_setup['num_units'] = num_units
    model_setup['dropout'] = dropout

    # Run the training
    trainer.run()

    # SageMaker saves the return value of train() in the `save` function in the resulting
    # model artifact model.tar.gz, and the contents of `output_data_dir` in the output
    # artifact output.tar.gz.

    # load the best model
    serializers.load_npz(os.path.join(output_data_dir, 'best_model.npz'), model)

    # remove the best model from output artifacts (since it will be saved as a model artifact)
    os.remove(os.path.join(output_data_dir, 'best_model.npz'))
    
    model_with_vocab_and_setup = (model, vocab, model_setup)
    
    save(model_with_vocab_and_setup, model_dir)
    
    return model, vocab, model_setup


def save(model, model_dir):
    
    trained_model, vocab, model_setup = model
    
    serializers.save_npz(os.path.join(model_dir, 'my_model.npz'), trained_model)
    with open(os.path.join(model_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
    with open(os.path.join(model_dir, 'args.json'), 'w') as f:
        json.dump(model_setup, f)



# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    This function loads models written during training into `model_dir`.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model

    For more on `model_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    model_path = os.path.join(model_dir, 'my_model.npz')

    vocab_path = os.path.join(model_dir, 'vocab.json')
    model_setup_path = os.path.join(model_dir, 'args.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    with open(model_setup_path, 'r') as f:
        model_setup = json.load(f)

    model_type = model_setup['model_type']
    if model_type == 'rnn':
        Encoder = nets.RNNEncoder
    elif model_type == 'cnn':
        Encoder = nets.CNNEncoder
    elif model_type == 'bow':
        Encoder = nets.BOWMLPEncoder
    num_layers = model_setup['num_layers']
    num_units = model_setup['num_units']
    dropout = model_setup['dropout']
    num_classes = model_setup['num_classes']
    encoder = Encoder(n_layers=num_layers, n_vocab=len(vocab), n_units=num_units, dropout=dropout)
    model = nets.TextClassifier(encoder, num_classes)

    serializers.load_npz(model_path, model)

    return model, vocab


def _npy_loads(data):
    """
    Deserializes npy-formatted bytes into a numpy array
    """
    stream = BytesIO(data)
    return np.load(stream)


def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


def input_fn(input_bytes, content_type):
    """This function is called on the byte stream sent by the client, and is used to deserialize the
    bytes into a Python object suitable for inference by predict_fn -- in this case, a NumPy array.
    
    This implementation is effectively identical to the default implementation used in the Chainer
    container, for NPY formatted data. This function is included in this script to demonstrate
    how one might implement `input_fn`.

    Args:
        input_bytes (numpy array): a numpy array containing the data serialized by the Chainer predictor
        content_type: the MIME type of the data in input_bytes
    Returns:
        a NumPy array represented by input_bytes.
    """
    if content_type == 'application/x-npy':
        return _npy_loads(input_bytes)
    else:
        raise ValueError('Content type must be application/x-npy')


def predict_fn(input_data, model):
    """
    This function receives a NumPy array and makes a prediction on it using the model returned
    by `model_fn`.
    
    The default predictor used by `Chainer` serializes input data to the 'npy' format:
    https://docs.scipy.org/doc/numpy-1.14.0/neps/npy-format.html

    The Chainer container provides an overridable pre-processing function `input_fn`
    that accepts the serialized input data and deserializes it into a NumPy array.
    `input_fn` is invoked before `predict_fn` and passes its return value to this function
    (as `input_data`)
    
    The Chainer container provides an overridable post-processing function `output_fn`
    that accepts this function's return value and serializes it back into `npy` format, which
    the Chainer predictor can deserialize back into a NumPy array on the client.

    Args:
        input_data: a numpy array containing the data serialized by the Chainer predictor
        model: the return value of `model_fn`
    Returns:
        a NumPy array containing predictions which will be returned to the client


    For more on `input_fn`, `predict_fn` and `output_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    trained_model, vocab = model

    words_batch = []
    for sentence in input_data.tolist():
        text = normalize_text(sentence)
        words = split_text(text)
        words_batch.append(words)

    xs = transform_to_array(words_batch, vocab, with_label=False)
    xs = convert_seq(xs, with_label=False)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        probs = trained_model.predict(xs, softmax=True)
    answers = trained_model.xp.argmax(probs, axis=1)
    scores = probs[trained_model.xp.arange(answers.size), answers].tolist()

    output = []
    for words, answer, score in zip(words_batch, answers, scores):
        output.append([' '.join(words), answer, score])

    return np.array(output)


def output_fn(prediction_output, accept):
    """This function is called on the return value of predict_fn, and is used to serialize the
    predictions back to the client.
    
    This implementation is effectively identical to the default implementation used in the Chainer
    container, for NPY formatted data. This function is included in this script to demonstrate
    how one might implement `output_fn`.

    Args:
        prediction_output (numpy array): a numpy array containing the data serialized by the Chainer predictor
        accept: the MIME type of the data expected by the client.
    Returns:
        a tuple containing a serialized NumPy array and the MIME type of the serialized data.
    """
    if accept == 'application/x-npy':
        return _npy_dumps(prediction_output), 'application/x-npy'
    else:
        raise ValueError('Accept header must be application/x-npy')
