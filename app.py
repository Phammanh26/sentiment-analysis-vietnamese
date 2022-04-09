from SentimentModel import  SentimentModel
from processing.CleanData import CleanData
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
from unittest import result
from flask import Flask, request, jsonify, render_template

VOCABSIZE =  30000
MAX_SEQUENCE_LENGTH = 70
print('setup and loading model')
model = SentimentModel(model_path = './model/my_model')
model.loading()
print('---Done setup and loading model----')

app = Flask(__name__)


@app.route('/sentiment-analyst/predict', methods=['POST'])
def predict():
    content = request.json
    text = content['text']
    clean_data = CleanData(text)
    text_clean = clean_data.preprocess_sentence()
    onehot_repr=[one_hot(words,VOCABSIZE)for words in [text_clean]]
    embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=MAX_SEQUENCE_LENGTH)
    
    X =np.array(embedded_docs)
    value = model.predict(X)[0][0]
    print("--------------")
    print(value)
    return jsonify(result = 1.*np.float32(value))

@app.route('/sentiment-analyst', methods=['GET', 'POST'])
def show1():
    temple ='sentiment-analyst.html'
    return render_template(temple)



    

if __name__ == '__main__':
    print('starting!!!!')
    app.run(host= '0.0.0.0',debug=True)
