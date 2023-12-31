
# Generated by CodiumAI

import pytest

class TestTokenizeText:

    #  The function should tokenize the text data correctly.
    def test_tokenize_text_tokenization(self):
        # Arrange
        X_train = ["This is a test sentence.", "Another test sentence."]
        X_test = ["Yet another test sentence."]
        max_words = 10000
        max_sequence_length = 200
    
        # Act
        X_train_seq, X_test_seq = tokenize_text(X_train, X_test, max_words, max_sequence_length)
    
        # Assert
        assert isinstance(X_train_seq, np.ndarray)
        assert isinstance(X_test_seq, np.ndarray)
        assert len(X_train_seq) == len(X_train)
        assert len(X_test_seq) == len(X_test)
        assert X_train_seq.shape[1] == max_sequence_length
        assert X_test_seq.shape[1] == max_sequence_length

    #  The function should pad the sequences correctly.
    def test_tokenize_text_padding(self):
        # Arrange
        X_train = ["This is a test sentence.", "Another test sentence."]
        X_test = ["Yet another test sentence."]
        max_words = 10000
        max_sequence_length = 200
    
        # Act
        X_train_seq, X_test_seq = tokenize_text(X_train, X_test, max_words, max_sequence_length)
    
        # Assert
        assert np.all(X_train_seq >= 0)
        assert np.all(X_test_seq >= 0)
        assert np.all(X_train_seq <= max_words)
        assert np.all(X_test_seq <= max_words)
        assert X_train_seq.shape[1] == max_sequence_length
        assert X_test_seq.shape[1] == max_sequence_length

    #  The function should return two sequences of equal length.
    def test_tokenize_text_equal_length(self):
        # Arrange
        X_train = ["This is a test sentence.", "Another test sentence."]
        X_test = ["Yet another test sentence."]
        max_words = 10000
        max_sequence_length = 200
    
        # Act
        X_train_seq, X_test_seq = tokenize_text(X_train, X_test, max_words, max_sequence_length)
    
        # Assert
        assert len(X_train_seq) == len(X_test_seq)

    #  The function should handle input strings that exceed the maximum sequence length.
    def test_tokenize_text_long_sequence(self):
        # Arrange
        X_train = ["This is a test sentence.", "Another test sentence."]
        X_test = ["Yet another test sentence that is longer than the maximum sequence length."]
        max_words = 10000
        max_sequence_length = 20
    
        # Act
        X_train_seq, X_test_seq = tokenize_text(X_train, X_test, max_words, max_sequence_length)
    
        # Assert
        assert X_train_seq.shape[1] == max_sequence_length
        assert X_test_seq.shape[1] == max_sequence_length

    #  The function should handle input strings that exceed the maximum number of words.
    def test_tokenize_text_long_words(self):
        # Arrange
        X_train = ["This is a test sentence.", "Another test sentence."]
        X_test = ["Yet another test sentence with more than the maximum number of words."]
        max_words = 5
        max_sequence_length = 200
    
        # Act
        X_train_seq, X_test_seq = tokenize_text(X_train, X_test, max_words, max_sequence_length)
    
        # Assert
        assert np.all(X_train_seq < max_words)
        assert np.all(X_test_seq < max_words)

    #  The function should handle input strings that contain non-ASCII characters.
    def test_tokenize_text_non_ascii(self):
        # Arrange
        X_train = ["This is a test sentence.", "Another test sentence."]
        X_test = ["Yet another test sentence with non-ASCII characters: éèàç."]
        max_words = 10000
        max_sequence_length = 200
    
        # Act
        X_train_seq, X_test_seq = tokenize_text(X_train, X_test, max_words, max_sequence_length)
    
        # Assert
        assert isinstance(X_train_seq, np.ndarray)
        assert isinstance(X_test_seq, np.ndarray)
        assert len(X_train_seq) == len(X_train)
        assert len(X_test_seq) == len(X_test)
        assert X_train_seq.shape[1] == max_sequence_length
        assert X_test_seq.shape[1] == max_sequence_length
