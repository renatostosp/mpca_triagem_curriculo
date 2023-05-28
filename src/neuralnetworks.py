# TROCAR CAMEL CASE POR SNAKE CASE: compileModel -> compile_model)
# trocar validation_split por validation_data (veja a rede CNN)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Bidirectional, Conv1D, LSTM
from keras.callbacks import ModelCheckpoint
from abc import ABC, abstractmethod
import os

class BuildModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compileModel(self):
        pass

    @abstractmethod
    def fitModel(self, X_train, y_train, epochs, callbacks):
        pass

    @abstractmethod
    def predictModel(self, X_test):
        pass

class Checkpoint:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_checkpoint_callback(self):
        return ModelCheckpoint(filepath=self.checkpoint_dir, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

class FeedForward(BuildModel):
    def __init__(self, max_len, num_classes):
        self.model = Sequential()
        self.model.add(Dense(units=256, input_shape=(max_len,), activation='relu'))
        self.model.add(Dense(units=num_classes, activation='softmax'))

    def compileModel(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def fitModel(self, X_train, y_train, X_val, y_val, epochs, callbacks=None):
        if callbacks is None:
            callbacks = []
        checkpoint = Checkpoint(f'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus\\models\\{self.__class__.__name__}')  
        checkpoint_callback = checkpoint.get_checkpoint_callback()
        callbacks.append(checkpoint_callback)
        self.callbacks = callbacks
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=self.callbacks)
        return history

    def predictModel(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluateModel(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)
    
class EmbFeedForward(BuildModel):

    def __init__(self, max_len, vocab_size, embedding_dim, num_classes):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        self.model.add(Flatten())
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=num_classes, activation='softmax'))

    def compileModel(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def fitModel(self, X_train, y_train, X_val, y_val, epochs, callbacks=None):
        if callbacks is None:
            callbacks = []
        checkpoint = Checkpoint(f'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus\\models\\{self.__class__.__name__}')  
        checkpoint_callback = checkpoint.get_checkpoint_callback()
        callbacks.append(checkpoint_callback)
        self.callbacks = callbacks
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=self.callbacks)
        return history

    def predictModel(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluateModel(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)


class BidirectionalLSTM(BuildModel):

    def __init__(self, max_len, vocab_size, embedding_dim, num_classes):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        self.model.add(Bidirectional(LSTM(32, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(32)))
        self.model.add(Dense(units=num_classes, activation='softmax'))

    def compileModel(self):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return self.model

    def fitModel(self, X_train, y_train, X_val, y_val, epochs, callbacks=None):
        if callbacks is None:
            callbacks = []
        checkpoint = Checkpoint(f'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus\\models\\{self.__class__.__name__}')  
        checkpoint_callback = checkpoint.get_checkpoint_callback()
        callbacks.append(checkpoint_callback)
        self.callbacks = callbacks
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=self.callbacks)
        return history

    def predictModel(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluateModel(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)


class CNN(BuildModel):

    def __init__(self, max_len, vocab_size, embedding_dim, num_classes):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        self.model.add(Conv1D(40, kernel_size=(3,3), activation='relu'))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Dense(units=num_classes, activation='softmax'))

    def compileModel(self):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return self.model

    def fitModel(self, X_train, y_train, X_val, y_val, epochs, callbacks=None):
        if callbacks is None:
            callbacks = []
        checkpoint = Checkpoint(f'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus\\models\\{self.__class__.__name__}')  
        checkpoint_callback = checkpoint.get_checkpoint_callback()
        callbacks.append(checkpoint_callback)
        self.callbacks = callbacks
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=self.callbacks)
        return history

    def predictModel(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluateModel(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)
