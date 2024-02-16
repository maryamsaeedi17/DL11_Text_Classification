import tensorflow as tf
import numpy as np
import pandas as pd

class EmojiTextClassifier():
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.words_vector = None
        self.model = None

    def read_csv(self, file_path):
        df = pd.read_csv(file_path)
        X = np.array(df["sentence"])
        Y = np.array(df["label"], dtype=int)

        return X, Y

    def label_to_emoji(self, label):
        emojies = ["❤️", "⚾️", "😄", "😔", "🍴"]
        return emojies[label]

    def load_dataset(self, dataset_path):
        self.X_train, self.Y_train = self.read_csv(f"{dataset_path}/train.csv")
        self.X_test, self.Y_test = self.read_csv(f"{dataset_path}/test.csv")

    def loade_feature_vectors(self, feature_file_path):
        f = open(feature_file_path, encoding="utf-8")

        self.words_vector = {}
        for line in f:
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:], dtype=np.float64)
            self.words_vector[word] = vector

        return self.words_vector

    def sentence_to_feature_vectors_ave(self, sentence, dim, words_vector):
        sentence = sentence.lower()
        words = sentence.strip().split(" ")

        sum_vectors = np.zeros((dim, ))
        for word in words:
            sum_vectors += words_vector[word]

        ave_vector = sum_vectors / len(words)

        return ave_vector

    def load_model(self, dim):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(5, input_shape=(dim, ), activation="Softmax")
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, dim, words_vector):
        X_train_ave = []

        for x_train in self.X_train:
            X_train_ave.append(self.sentence_to_feature_vectors_ave(x_train, dim, words_vector))

        X_train_ave = np.array(X_train_ave)

        Y_train_one_hoted = tf.keras.utils.to_categorical(self.Y_train, num_classes=5)

        self.model.fit(X_train_ave, Y_train_one_hoted, epochs=200)

    def test(self, dim, words_vector):
        X_test_ave = []

        for x_test in self.X_test:
            X_test_ave.append(self.sentence_to_feature_vectors_ave(x_test, dim, words_vector))

        X_test_ave = np.array(X_test_ave)

        Y_test_one_hoted = tf.keras.utils.to_categorical(self.Y_test, num_classes=5)
        
        self.model.evaluate(X_test_ave, Y_test_one_hoted)

    def predict(self, my_test_sentence, dim, words_vector):
        test_ave = self.sentence_to_feature_vectors_ave(my_test_sentence, dim, words_vector)
        test_ave = np.array([test_ave])
        result = self.model.predict(test_ave)
        y_pred = np.argmax(result)
        return(self.label_to_emoji(y_pred))
