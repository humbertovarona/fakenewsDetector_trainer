def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64, learning_rate=0.001, enable_gpu=False, verbose=True):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    if enable_gpu:
        with tf.device('/GPU:0'):
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0 if not verbose else 1)
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0 if not verbose else 1)
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy
