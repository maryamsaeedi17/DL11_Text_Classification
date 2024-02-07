

class EmojiTextClassifier():
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.words_vector = None

    def read_csv(self, file_path):
        df = pd.read_csv(file_path)
        X = np.array(df["sentence"])
        Y = np.array(df["label"], dtype=int)

        return X, Y

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

    def sentence_to_feature_vectors_avg(self, sentence, dim, words_vector):
        sentence = sentence.lower()
        words = sentence.strip().split(" ")

        sum_vectors = np.zeros((dim, ))
        for word in words:
            sum_vectors += words_vector[word]

        ave_vector = sum_vectors / len(words)

        return ave_vector

    def load_model(self):
        ...

    def train(self):
        ...

    def test(self):
        ...

    def predict(self):
        ...

