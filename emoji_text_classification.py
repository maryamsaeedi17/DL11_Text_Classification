# -*- coding: utf-8 -*-
"""emoji_text_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I8GtUvYHY4F00R8bgBGTc7zN4qSPlXTv
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time

# !wget https://nlp.stanford.edu/data/glove.6B.zip

# !unzip -q glove.6B.zip -d /content/drive/MyDrive/datasets/glove.6B

from emoji_text_classifier import EmojiTextClassifier

etc = EmojiTextClassifier()

dim = 300

words_vector = etc.loade_feature_vectors(f"/content/drive/MyDrive/datasets/glove.6B/glove.6B.{dim}d.txt")

etc.load_dataset("/content/drive/MyDrive/datasets/Emoji_Text_Classification (1)")

etc.load_model(dim)

etc.train(dim, words_vector)

etc.test(dim, words_vector)

etc.predict("I am not interested in sweet food", dim, words_vector)

test_sentences, _ = etc.read_csv("/content/drive/MyDrive/datasets/Emoji_Text_Classification (1)/test.csv")
n = len(test_sentences)

start = time.time()
for test_sentence in test_sentences:
  etc.predict(test_sentence, dim, words_vector)

infrence_time = (time.time() - start) / n
print("Infrence time: ", infrence_time)