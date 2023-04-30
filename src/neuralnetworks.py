from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from tensorflow.keras import Input
from abc import ABC, abstractmethod
from typing import Optional


class BuildModel(ABC):

    @abstractmethod
    def compileModel(self):
        pass

    @abstractmethod
    def fitModel(self, X_train, y_train, epochs, callbacks):
        pass
 
    @abstractmethod
    def predictModel(self, X_test):
        pass

class MLP(BuildModel):

    def __init__(self, num_classes):
        super().__init__()
        self.model = Sequential([
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])

    def compileModel(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def fitModel(self, X_train, y_train, epochs, callbacks):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, callbacks=callbacks)    
        return history

    def predictModel(self, X_test):
        return self.model.predict(X_test, batch_size=64, verbose=1)
    
    def evaluateModel(self, X_test,y_test):
        return self.model.evaluate(X_test, y_test, batch_size=64, verbose=1)     

class BidirectionalLSTM(BuildModel):

    def __init__(self, num_classes):
        super().__init__()
        self.max_features = 20000
        # inputs = Input(shape=(None,), dtype="int32")
        # x = Embedding(num_classes, 128)(inputs)
        # x = Bidirectional(LSTM(64, return_sequences=True))(x)
        # x = Bidirectional(LSTM(64))(x)
        # outputs = Dense(num_classes, activation="sigmoid")(x)
        # self.model = Model(inputs, outputs)

        self.model = Sequential([
            Embedding(self.max_features, 128),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(num_classes, activation="sigmoid")
        ])

    def compileModel(self):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def fitModel(self, X_train, y_train, epochs):
        history = self.model.fit(X_train, y_train, batch_size=16, epochs=epochs, validation_split=0.1)
        return history

    def predictModel(self, X_test):
        return self.model.predict(X_test, batch_size=16, verbose=1)
    
    def evaluateModel(self, X_test,y_test):
        return self.model.evaluate(X_test, y_test, batch_size=16, verbose=1)    