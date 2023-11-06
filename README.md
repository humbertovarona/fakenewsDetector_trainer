# fakenewsDetector_trainer

Train models for the detection of fake news through 3 artificial neural network architectures

# Version

![](https://img.shields.io/badge/Version%3A-1.0-success)

# Release date

![](https://img.shields.io/badge/Release%20date-Jan%2C%206%2C%202023-9cf)

# License

![](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)

# Programming language

<img src="https://img.icons8.com/?size=512&id=13441&format=png" width="50"/>

# OS

<img src="https://img.icons8.com/?size=512&id=17842&format=png" width="50"/> <img src="https://img.icons8.com/?size=512&id=122959&format=png" width="50"/> <img src="https://img.icons8.com/?size=512&id=108792&format=png" width="50"/>

# Requirements

```bash
pip install tensorflow pandas sklearn
```

```python
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D, Reshape
from tensorflow.keras.regularizers import l2
```

# How to run

```python
filename = 'training_database.csv'
data = load_data(filename)
X_train, X_test, y_train, y_test = preprocess_data(data)
X_train_seq, X_test_seq = tokenize_text(X_train, X_test)
model = build_simplemodel(max_words=10000, max_sequence_length=200)
loss, accuracy = train_and_evaluate_model(model, X_train_seq, y_train, X_test_seq, y_test)
```

# How to save models

```python
model_filename =  'fake_or_real_news_simplemodel'
save_model(model, model_filename)
```

or

```python
model_filename =  'fake_or_real_news_simplemodel.h5'
save_model(model, model_filename)
```

# 'training_database.csv' file structure

It is made up of three columns separated by commas.

Samples:
```csv
Newsheadline, News, FakeOrReal
headline_1, news_1, fake
headline_2, news_2, fake
headline_3, news_3, real
headline_4, news_4, real
headline_5, news_5, fake
.
.
.
headline_n, news_n, fake 
```
The third column is a binary value, it can be [0,1], [fake,real] or [true,false]

# train_and_evaluate_model: Function parameters

Definition:
```python
loss, accuracy = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64, learning_rate=0.001, enable_gpu=False, verbose=True)
```

> Input arguments:
>
>> `model`: The Keras model to be trained and evaluated. It should be a model previously defined using the Keras library.
>>
>> `X_train`: The training data (features) to be used for training the model. It should be a NumPy array or a compatible data structure.
>>
>> `y_train`: The training labels corresponding to the training data. It should be a NumPy array or a compatible data structure.
>>
>> `X_test`: The test data (features) to be used for evaluating the model. It should be a NumPy array or a compatible data structure.
>>
>> `y_test`: The test labels corresponding to the test data. It should be a NumPy array or a compatible data structure.
>>
>> `epochs`: The number of epochs (complete iterations through the entire training dataset) to be used for training the model. It is set to 5 epochs by default.
>>
>> `batch_size`: The batch size to be used during training. It determines how many examples are processed at once before updating the model's weights. It is set to 64 by default.
>>
>> `learning_rate`: The learning rate that controls the rate at which the model's weights are adjusted during training. It is set to 0.001 by default.
>>
>> `enable_gpu`: A boolean flag that determines whether GPU (Graphics Processing Unit) training will be enabled if available. If True, training is performed on the GPU; otherwise, it's done on the CPU. It is set to False by default.
>>
>> `verbose`: A boolean flag that controls the verbosity during training and evaluation. If True, progress messages are displayed; if False, no messages are shown. It is set to True by default.
>
> Returns:
>
>> `loss`: This is a scalar value representing the loss (typically binary cross-entropy or another loss function) calculated during the evaluation of the model on the test data. The loss measures how well the model's predictions match the true labels, and a lower loss indicates better performance.
>>
>> `accuracy`: This is a scalar value representing the accuracy of the model on the test data. It is a measure of the proportion of correctly classified instances in the test dataset. It is typically expressed as a percentage, where higher values indicate better model performance.

# build_simplemodel, build_robustmodel, and build_ultrarobustmodel: Function parameters 

Definition:
```python
model = build_simplemodel(max_words=10000, max_sequence_length=200)

model = build_robustmodel(max_words=10000, max_sequence_length=200)

model = build_ultrarobustmodel(max_words=10000, max_sequence_length=200)
```

> Input arguments:
>
>> `max_words`: An integer that represents the maximum number of words in the vocabulary or the size of the input word embedding. This parameter determines the input dimension of the embedding layer.
>>
>> `max_sequence_length`: An integer that represents the maximum sequence length or the maximum number of time steps for input sequences. This parameter specifies the input length of the sequences to be processed by the model.
>
> Returns:
>
>> `model`: A Keras model that is constructed according to the provided specifications. The model consists of an embedding layer with max_words input dimension, followed by an LSTM layer with 128 units, and a final dense layer with a sigmoid activation function. The model is compiled using the Adam optimizer and binary cross-entropy loss, with accuracy as a metric.


# Model architecture

## build_simplemodel architecture

![build_simplemodel](images/fake_or_real_news_simplemodel.png)

## build_robustmodel architecture

![build_robustmodel](images/fake_or_real_news_robustmodel.png)

## build_ultrarobustmodel architecture

![build_ultrarobustmodel](images/fake_or_real_news_ultrarobustmodel.png)

