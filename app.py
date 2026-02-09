from flask import Flask,render_template,request,jsonify
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


try:
    model = joblib.load('iris_model')
except Exception as e:
    print(f"Error loading the model: {e}")



@app.route("/",methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/",methods=['POST'])
def predict():

    
    try:
        float_features = [float(x) for x in request.form.values()]
    
        features = [np.array(float_features)]
        prediction = model.predict(features)

        if prediction[0]==0:
            species="Setosa"
        elif  prediction[0]==1:
            species="Versicolor"
        else:
            species="Virginica"
            

        return render_template("index.html", prediction_text=f"{species:}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        return render_template("index.html", prediction_text="Error making prediction.")

if __name__ == "__main__":
    app.run(debug=True)