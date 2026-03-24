from flask import Flask , request , jsonify
from joblib import load
import pandas as pd
from DATA_PROCESSING_PIPELINE_CLASSES import *

app = Flask(__name__)

@app.route('/')
def healthpoint():
    return jsonify({"status" : "active"})

@app.route('/predict_default' , methods = ['POST'])
def predict_default():

    pipeline = load('Data-Pipeline.pkl')
    model = load('XGBClassifierFinal.pkl')
    df_cols = load('DF_COLS.pkl')
    
    feature_data = request.json
    df = pd.DataFrame(feature_data)
    df = df.reindex(columns = df_cols)

    
    df_transformed = pipeline.transform(df)
    predictions = list(model.predict(df_transformed))
    return jsonify({'prediction' : str(predictions)})

if __name__ == '__main__':
    app.run(host = "0.0.0.0" , port = 5000 , debug = False)