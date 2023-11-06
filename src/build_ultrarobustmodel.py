def build_ultrarobustmodel(max_words, max_sequence_length, verbose=True):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Reshape((-1, 256)))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if not verbose:
        model.callbacks = [keras.callbacks.TensorBoard(write_graph=False, log_dir='logs')]
    return model
