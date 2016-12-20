## Convolutional Neural Networks for Sentence Classification

This is Chainer based implemantation for [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence).

### Requirements
Code is written in Python 3.5 and requires Chainer and Common-ML.

Using the pre-trained `word2vec` vectors will also require downloading the binary file from [here](https://code.google.com/p/word2vec/).

Install the following modules:

```
pip install chainer
pip install common-ml
```

### Data Preprocessing

To process the raw data, run

```
python main.py vectorize
```

### Running the models

To process cross validation, run
```
python main.py cv
```

### Configuration file

See config.xml.


