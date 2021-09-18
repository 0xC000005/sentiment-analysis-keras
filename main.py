import json
import keras
import keras_preprocessing
import numpy as np
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = keras_preprocessing.text.tokenizer_from_json(data)

model = keras.models.load_model('model')


def perform_sentimental_analysis(word):
    twt = word
    # vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences(twt)
    # padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
    # print(twt)
    sentiment = model.predict(twt, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0):
        # print("negative")
        dict = {
            "word": str(word),
            "pos_acc": float(sentiment[1]),
            "neg_acc": float(sentiment[0]),
            "positive_result": False,
        }
        dict = json.dumps(dict)
        return dict
    elif (np.argmax(sentiment) == 1):
        # print("positive")
        dict = {
            "word": str(word),
            "pos_acc": float(sentiment[1]),
            "neg_acc": float(sentiment[0]),
            "positive_result": True,
        }
        dict = json.dumps(dict)
        return dict

    # print("pos_acc: " + str(sentiment[1]) + "% \nneg_acc: " + str(sentiment[0]) + "%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_sentence = ['After', 'using', 'the', 'Unfold', 'AI','therapy', 'I', 'feel', 'much', 'better', 'now']
    ans = []
    for word in test_sentence:
        ans.append(perform_sentimental_analysis(word))
    for result in ans:
        print(result)
