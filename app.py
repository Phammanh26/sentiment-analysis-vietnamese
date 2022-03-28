from SentimentModel import  SentimentModel
from processing.CleanData import CleanData
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
from flask import Flask, request, jsonify

VOCABSIZE =  30000
MAX_SEQUENCE_LENGTH = 70
model = SentimentModel(model_path = './model/my_model')
model.loading()

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    content = request.json
    text = content['text']
    clean_data = CleanData(text)
    text_clean = clean_data.preprocess_sentence()
    onehot_repr=[one_hot(words,VOCABSIZE)for words in [text_clean]]
    embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=MAX_SEQUENCE_LENGTH)
    
    X =np.array(embedded_docs)
    value = model.predict(X)[0]
    
    if value > 0.5:
        return jsonify({'result:': "day la mot cau tích cực: {}".format(value)})
    else:
        return jsonify({'result:': "day la mot cau tiêu cực: {}".format(value)})
    

if __name__ == '__main__':
    print('fuck')
    app.run(host= '0.0.0.0',debug=True)
