# TROCAR CAMEL CASE POR SNAKE CASE: compile_model -> compile_model)
# trocar validation_split por validation_data (veja a rede CNN)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Bidirectional, Conv1D, LSTM, GlobalMaxPool1D
from keras.callbacks import ModelCheckpoint
from abc import ABC, abstractmethod
import os

class Checkpoint:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_checkpoint_callback(self):
        return ModelCheckpoint(filepath=self.checkpoint_dir, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True) 

class BuildModel(ABC):
    @abstractmethod
    def __init__(self)  -> Sequential:
        pass

    @abstractmethod
    def fit_model(self, X_train, y_train, epochs, callbacks):
        pass

    @abstractmethod
    def predict_model(self, X_test):
        pass

class FeedForward(BuildModel):
    def __init__(self, max_len: int, num_classes: int) -> None:
        self.cb = False
        self.model = Sequential()
        self.model.add(Dense(units=256, input_shape=(max_len,), activation='relu'))
        self.model.add(Dense(units=num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs):
        callbacks_list = kwargs.get('callbacks', [])        
        if 'checkpoint_dir' in kwargs and not callbacks_list:
            checkpoint = Checkpoint(f"{kwargs['checkpoint_dir']}")
            checkpoint_callback = checkpoint.get_checkpoint_callback()
            callbacks_list.append(checkpoint_callback)
        else:
            self.cb = True                            
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list)
        return history

    def predict_model(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)
    
    
class EmbFeedForward(BuildModel):

    def __init__(self, vocab_size: int, max_len: int, num_classes: int, embedding_dim: int):
        self.cb = False
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        self.model.add(Flatten())
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs):
        callbacks_list = kwargs.get('callbacks', [])        
        if 'checkpoint_dir' in kwargs and not callbacks_list:
            checkpoint = Checkpoint(f"{kwargs['checkpoint_dir']}{self.__class__.__name__}")
            checkpoint_callback = checkpoint.get_checkpoint_callback()
            callbacks_list.append(checkpoint_callback)
        else:
            self.cb = True                            
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list)
        return history

    def predict_model(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)


class BidirectionalLSTM(BuildModel):

    def __init__(self, vocab_size: int, max_len: int, num_classes: int, embedding_dim: int):
        self.cb = False
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        self.model.add(Bidirectional(LSTM(32, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(32)))
        self.model.add(Dense(units=num_classes, activation='softmax'))
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def fit_model(self, X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs):
        callbacks_list = kwargs.get('callbacks', [])        
        if 'checkpoint_dir' in kwargs and not callbacks_list:
            checkpoint = Checkpoint(f"{kwargs['checkpoint_dir']}{self.__class__.__name__}")
            checkpoint_callback = checkpoint.get_checkpoint_callback()
            callbacks_list.append(checkpoint_callback)
        else:
            self.cb = True                            
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list)
        return history

    def predict_model(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)


class CNN(BuildModel):

    def __init__(self, vocab_size: int, max_len: int, num_classes: int, embedding_dim: int, num_filters: int,
                    kernel_size: int):
        self.cb = False
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        self.model.add(Conv1D(num_filters, kernel_size, activation='relu'))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=num_classes, activation='sigmoid'))
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def fit_model(self, X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs):
        callbacks_list = kwargs.get('callbacks', [])       
        if 'checkpoint_dir' in kwargs and not callbacks_list:
            checkpoint = Checkpoint(f"{kwargs['checkpoint_dir']}{self.__class__.__name__}")
            checkpoint_callback = checkpoint.get_checkpoint_callback()
            callbacks_list.append(checkpoint_callback)
        else:
            self.cb = True        
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list)
        return history

    def predict_model(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)
