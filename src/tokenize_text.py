def tokenize_text(X_train, X_test, max_words=10000, max_sequence_length=200):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_seq = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_test_seq = pad_sequences(X_test_seq, maxlen=max_sequence_length)
    return X_train_seq, X_test_seq
