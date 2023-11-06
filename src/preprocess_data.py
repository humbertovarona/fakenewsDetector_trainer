def preprocess_data(data):
    X = data['Newsheadline'] + ' ' + data['News']
    y = data['FakeOrReal']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
