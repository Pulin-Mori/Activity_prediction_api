from flask import Flask,request,jsonify
import numpy as np
import pickle

app=Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return "Hello WorldD"

@app.route('/predict', methods = ['POST'])
def predict():
    result=0
    gx = request.form.get('Gyr_x')
    gy = request.form.get('Gyr_y')
    gz = request.form.get('Gyr_z')
    ax = request.form.get('Acc_x')
    ay = request.form.get('Acc_y')
    az = request.form.get('Acc_z')

    input = np.array([[gx,gy,gz,ax,ay,az]])
    result = model.predict(input)

    return jsonify({"Activity":str(result)})

if __name__=='__main__':
    app.run(debug=True)