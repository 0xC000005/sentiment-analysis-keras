import json

import keras_preprocessing
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = keras_preprocessing.text.tokenizer_from_json(data)

model = keras.models.load_model('model')


def perform_sentimental_analysis(word):
    twt = [word]
    # vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences(twt)
    # padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
    # print(twt)
    sentiment = model.predict(twt, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0):
        # print("negative")
        dict = {
            "word": word,
            "pos_acc": sentiment[1],
            "neg_acc": sentiment[0],
            "result": "negative",
        }
        return dict
    elif (np.argmax(sentiment) == 1):
        # print("positive")
        dict = {
            "word": word,
            "pos_acc": sentiment[1],
            "neg_acc": sentiment[0],
            "result": "positive",
        }
        return dict

    # print("pos_acc: " + str(sentiment[1]) + "% \nneg_acc: " + str(sentiment[0]) + "%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_sentence = ['I','had', 'a', 'good', 'day']
    ans = []
    for word in test_sentence:
        ans.append(perform_sentimental_analysis(word))
    print(ans)