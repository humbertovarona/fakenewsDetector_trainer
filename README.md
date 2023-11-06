# fakenewsDetector_trainer

Train models for the detection of fake new through 3 artificial neural network architectures

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

# 'training_database.csv' file structure

```csv
```

