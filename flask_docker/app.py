
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os


app = Flask(__name__)

pipeline = joblib.load( './model/' +  os.listdir('./model/')[0])

@app.route('/predict',methods=['POST'])
def predict():
    
    data = request.get_json(force=True)
    data = pd.DataFrame([data.values()], columns= data.keys())

    prediction = pipeline.predict_proba(data)[0][-1]
    # print(prediction)
    output = str(round(prediction*100,2))
    print(output)
    # output = 'a'
    return jsonify({'Probability of fraud': f'% {output}'})


if __name__ == '__main__':
    app.run(debug=True)