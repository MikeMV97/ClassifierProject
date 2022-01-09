import joblib
import numpy as np
import pandas as pd
from featuresHelper import FeaturesHelper
import json

from flask import Flask, Response, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    featHelper = FeaturesHelper()
    article_text = pd.Series(np.array([request_data['article_text']]))
    features = featHelper.add_features(article_text)
    prediction = model.predict(features)
    features_list = features.values.tolist()[0]

    js = {  'avg_word_len': features_list[1], 
            'sentiment_txt': features_list[2], 
            'num_words': features_list[3],
            'num_diff_words': features_list[4],
            'num_stopwords': features_list[5],
            'rate_stopwords_words': features_list[6],
            'rate_diffwords_words': features_list[7],
            'prediction_result': int(prediction[0])
        }
        
    return Response(json.dumps(js), mimetype='application/json')


if __name__ == "__main__":
    model = joblib.load('./models/SVC_model_0.8342.pkl')
    app.run(port=8080)