
# Generated by CodiumAI

import pytest

class TestBuildSimplemodel:

    #  The function builds a sequential model with an embedding layer, LSTM layer, and dense layer with sigmoid activation.
    def test_build_model_with_embedding_lstm_dense(self):
        # Arrange
        max_words = 1000
        max_sequence_length = 50
    
        # Act
        model = build_simplemodel(max_words, max_sequence_length)
    
        # Assert
        assert isinstance(model, Sequential)
        assert len(model.layers) == 3
        assert isinstance(model.layers[0], Embedding)
        assert isinstance(model.layers[1], LSTM)
        assert isinstance(model.layers[2], Dense)
        assert model.layers[2].activation == 'sigmoid'
        assert model.loss == 'binary_crossentropy'
        assert model.optimizer == 'adam'
        assert model.metrics == ['accuracy']

    #  The model is compiled with binary crossentropy loss and accuracy metric using the Adam optimizer.
    def test_compile_model(self):
        # Arrange
        max_words = 1000
        max_sequence_length = 50
    
        # Act
        model = build_simplemodel(max_words, max_sequence_length)
    
        # Assert
        assert model.loss == 'binary_crossentropy'
        assert model.optimizer == 'adam'
        assert model.metrics == ['accuracy']

    #  The function returns the compiled model.
    def test_return_compiled_model(self):
        # Arrange
        max_words = 1000
        max_sequence_length = 50
    
        # Act
        model = build_simplemodel(max_words, max_sequence_length)
    
        # Assert
        assert isinstance(model, Sequential)

    #  max_words and max_sequence_length are both 0.
    def test_zero_max_words_and_sequence_length(self):
        # Arrange
        max_words = 0
        max_sequence_length = 0
    
        # Act
        model = build_simplemodel(max_words, max_sequence_length)
    
        # Assert
        assert isinstance(model, Sequential)

    #  max_words and max_sequence_length are both very large.
    def test_large_max_words_and_sequence_length(self):
        # Arrange
        max_words = 1000000
        max_sequence_length = 100000
    
        # Act
        model = build_simplemodel(max_words, max_sequence_length)
    
        # Assert
        assert isinstance(model, Sequential)

    #  max_words is negative.
    def test_negative_max_words(self):
        # Arrange
        max_words = -1000
        max_sequence_length = 50
    
        # Act and Assert
        with pytest.raises(ValueError):
            build_simplemodel(max_words, max_sequence_length)
