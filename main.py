# coding: utf-8

from logging import getLogger
import pickle
import re
import sys

from chainer import optimizers, Chain, Variable
from commonml.runner import runner
from commonml.skchainer import ChainerEstimator, \
    SoftmaxCrossEntropyClassifier, XyDataset
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection._split import KFold

import chainer.functions as F
import chainer.links as L
import numpy as np


logger = getLogger('main')


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"'", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, k=300):
    unknown = 0
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
            logger.info('%s is not found in word2vec.', word)
            unknown += 1
    logger.info('%d words are not found in word2vec.', unknown)


def vectorize(config):
    file_encoding = config.get('file_encoding')

    datasets = []
    targets = []
    vocab = {"": 0}

    def process_text_file(path, y):
        max_len = 0
        with open(path, 'r', encoding=file_encoding) as f:
            for line in f:
                normalized_line = clean_str(line)
                targets.append(y)
                word_list = normalized_line.split()
                if max_len < len(word_list):
                    max_len = len(word_list)
                words = set(word_list)
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
                wordid_list = []
                for word in word_list:
                    wordid_list.append(vocab[word])
                datasets.append(wordid_list)
        return max_len

    max_doc_len = 0
    for pos_file in config.get('pos_files'):
        length = process_text_file(pos_file, 1)
        if length > max_doc_len:
            max_doc_len = length
    for neg_file in config.get('neg_files'):
        length = process_text_file(neg_file, 0)
        if length > max_doc_len:
            max_doc_len = length

    for data_list in datasets:
        while len(data_list) < max_doc_len:
            data_list.append(0)

    data_file = config.get('data_file')
    pickle.dump([np.array(datasets, dtype=np.int32), np.array(targets, dtype=np.int32), vocab], open(data_file, "wb"))
    logger.info('Saving data file: %s', data_file)


def cv(config):
    np.random.seed(3435)

    data_file = config.get('data_file')
    with open(data_file, "rb") as f:
        (datasets, targets, vocab) = pickle.load(f)
    logger.info('Loaded vect_file: %s', data_file)

    if config.get('vector_type') == 'word2vec':
        w2v_file = config.get('w2v_file')
        w2v = load_bin_vec(w2v_file, vocab)
        add_unknown_words(w2v, vocab)
        initialW = []
        for entry in sorted(vocab.items(), key=lambda x: x[1]):
            initialW.append(w2v[entry[0]])
        initialW = np.array(initialW)
        logger.info('Loaded word2vec: %s', w2v_file)
    else:
        initialW = None

    model_config = {}
    model_config.update(config.get('model'))
    model_config['batch_size'] = config.get('batch_size')
    model_config['epoch'] = config.get('epoch')
    model_config['gpu'] = config.get('gpu')
    model_config['non_static'] = config.get('non_static')
    model_config['n_vocab'] = len(vocab)
    model_config['doc_length'] = datasets.shape[1]
    model_config['initialW'] = initialW
    for phase in range(1, config.get('phase') + 1):
        logger.info('Cross Validation: %d/%d', phase, config.get('phase'))
        kf = KFold(n_splits=config.get('split'))
        for train_index, test_index in kf.split(datasets):
            train_index = np.random.permutation(train_index)
            X_train = datasets[train_index]
            Y_train = targets[train_index]
            X_test = datasets[test_index]
            Y_test = targets[test_index]

            logger.info('Fitting: %s -> %s', X_train.shape, Y_train.shape)
            (_, clf) = create_classifier(**model_config)
            clf.fit(X_train, Y_train,
                    dataset_creator=lambda X, y, model: XyDataset(X=X, y=y, model=model, X_dtype=np.int32))

            logger.info('Predicting: %s -> %s', X_test.shape, Y_test.shape)
            preds = clf.predict(X_test,
                                dataset_creator=lambda X, model: XyDataset(X=X, model=model, X_dtype=np.int32))

            logger.info('accuracy: {0}'.format(accuracy_score(Y_test, preds)))
            if config.get('fold_out'):
                break

    logger.info('Done')


def create_classifier(n_vocab, doc_length, wv_size, filter_sizes, hidden_units, output_channel, initialW, non_static, batch_size, epoch, gpu):
    model = NNModel(n_vocab=n_vocab,
                    doc_length=doc_length,
                    wv_size=wv_size,
                    filter_sizes=filter_sizes,
                    hidden_units=hidden_units,
                    output_channel=output_channel,
                    initialW=initialW,
                    non_static=non_static)
#    optimizer = optimizers.Adam()
    optimizer = optimizers.AdaDelta()
    return (model, ChainerEstimator(model=SoftmaxCrossEntropyClassifier(model),
                                    optimizer=optimizer,
                                    batch_size=batch_size,
                                    device=gpu,
                                    stop_trigger=(epoch, 'epoch')))


class NNModel(Chain):

    def __init__(self, n_vocab, doc_length, wv_size, filter_sizes=[3, 4, 5], hidden_units=[100, 2], output_channel=100, initialW=None, non_static=False):
        super(NNModel, self).__init__()
        self.filter_sizes = filter_sizes
        self.hidden_units = hidden_units
        self.doc_length = doc_length
        self.non_static = non_static

        self.add_link('embed', F.EmbedID(n_vocab, wv_size, initialW=initialW, ignore_label=0))
        for filter_h in self.filter_sizes:
            filter_w = wv_size
            filter_shape = (filter_h, filter_w)
            self.add_link('conv' + str(filter_h), L.Convolution2D(1, output_channel, filter_shape))

        for i in range(len(hidden_units)):
            self.add_link('l' + str(i), L.Linear(None, hidden_units[i]))

    def __call__(self, x, train=True):
        hlist = []
        h_0 = self['embed'](x)
        if not self.non_static:
            h_0 = Variable(h_0.data)
        h_1 = F.reshape(h_0, (h_0.shape[0], 1, h_0.shape[1], h_0.shape[2]))
        for filter_h in self.filter_sizes:
            pool_size = (self.doc_length - filter_h + 1, 1)
            h = F.max_pooling_2d(F.relu(self['conv' + str(filter_h)](h_1)), pool_size)
            hlist.append(h)
        h = F.concat(hlist)
        pos = 0
        while pos < len(self.hidden_units) - 1:
            h = F.dropout(F.relu(self['l' + str(pos)](h)))
            pos += 1
        y = F.relu(self['l' + str(pos)](h))
        return y


if __name__ == '__main__':
    sys.exit(runner.run())
