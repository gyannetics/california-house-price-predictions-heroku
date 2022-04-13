from flask import Flask, render_template, request, jsonify
from pandas import DataFrame
# from sklearn.externals import joblib
import joblib
import traceback
# from os import Path

app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction(): 
    if model:
        try:
            json = request.json
            prediction = list(model.predict(DataFrame(json)))
            return jsonify({'prediction': str(prediction)})    
        except Exception as e:
            return jsonify({'trace': traceback.format_exc()})
     #render_template('submit.html', prediction=prediction)

# @app.route('/submit', methods=['POST'])
# def submit(): 
#     # HTML to PYTHON
#     if request.method == 'POST':
#         name = request.form['username']

#     # PYTHON TO HTML
#     return render_template('submit.html', n=name)

if __name__ == '__main__':
    model = joblib.load("./saved_model/model.pkl") # Load "model.pkl"
    print ('Model loaded')
    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')

    # app.run(port=port, debug=True)    
    app.run(debug=True)