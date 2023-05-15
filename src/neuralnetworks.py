from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding, Flatten, Conv1D
from keras import Input
from abc import ABC, abstractmethod


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
        self.vocab_size = 300 
        self.embedding_dim = 100
        self.drop_value = 0.5
        self.n_dense = 24
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=1000),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(self.drop_value),
            Dense(64, activation="relu"),
            Dropout(self.drop_value),
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
        self.max_features = 30000
        # inputs = Input(shape=(None,), dtype="int32")
        # x = Embedding(num_classes, 128)(inputs)
        # x = Bidirectional(LSTM(64, return_sequences=True))(x)
        # x = Bidirectional(LSTM(64))(x)
        # outputs = Dense(num_classes, activation="sigmoid")(x)
        # self.model = Model(inputs, outputs)

        self.model = Sequential([
            Input(shape=(None,), dtype="int32"),
            Embedding(self.max_features, 64),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(num_classes, activation="softmax")
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


class CNN(BuildModel):

    def __init__(self, num_classes):
        super().__init__()
        self.max_features = 30000
        # inputs = Input(shape=(None,), dtype="int32")
        # x = Embedding(num_classes, 128)(inputs)
        # x = Bidirectional(LSTM(64, return_sequences=True))(x)
        # x = Bidirectional(LSTM(64))(x)
        # outputs = Dense(num_classes, activation="sigmoid")(x)
        # self.model = Model(inputs, outputs)

        self.model = Sequential([
            Input(shape=(None,), dtype="int32"),
            Embedding(self.max_features, 64),
            Conv1D(40, kernel_size=(3,3), activation='relu'),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(num_classes, activation="softmax")
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