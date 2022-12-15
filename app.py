
from flask import Flask, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)
pipeline = joblib.load("./models/current_model/model_20221215_1547.p")

@app.route('/predict',methods=['POST'])
def predict(pipeline):
    
    data = request.get_json(force=True)
    data = pd.DataFrame([data.values()], columns= data.keys())

    prediction = pipeline.predict_proba(data)[0][-1]
    # print(prediction)
    output = str(round(prediction*100,2))
    print(output)
    # output = 'a'
    return jsonify({'Probability of fraud': f'% {output}'})

if __name__ == '__main__':
    app.run(port=5001, debug=True)