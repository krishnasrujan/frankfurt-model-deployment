import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('frankfurt_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    High,Open,Close=int_features[0],int_features[1],int_features[2]
    HL_PCT=(High-Close)/Close*100
    PCT_change=(Close-Open)/Open*100
    final_features = [np.array([Close,HL_PCT,PCT_change])]
    prediction = model.predict(final_features)


    return render_template('index.html', prediction_text='Estimated closing stock price is  $ {}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)

