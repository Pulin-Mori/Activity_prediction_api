from flask import Flask,request,jsonify
import numpy as np
import pickle

app=Flask(__name__)

trained_model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    # return jsonify({"Hello World":"1"})
    ax = request.args.get("Acc_x")
    ay = request.args.get("Acc_y")
    az = request.args.get("Acc_z")
    gx = request.args.get("Gyr_x")
    gy = request.args.get("Gyr_y")
    gz = request.args.get("Gyr_z")

    input = np.array([[gx,gy,gz,ax,ay,az]])
    result = trained_model.predict(input).tolist()[0]

    return jsonify({"Activity":str(result),"ax":str(ax)})

@app.route('/predict', methods = ['GET','POST'])
def predict():
    gx = request.form.get('Gyr_x')
    gy = request.form.get('Gyr_y')
    gz = request.form.get('Gyr_z')
    ax = request.form.get('Acc_x')
    ay = request.form.get('Acc_y')
    az = request.form.get('Acc_z')

    input = np.array([[gx,gy,gz,ax,ay,az]])
    result = trained_model.predict(input).tolist()[0]

    return jsonify({"Activity":str(result)})

if __name__=='__main__':
    app.run(debug=True)