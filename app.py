from flask import Flask,request,jsonify
import numpy as np
import pickle

app=Flask(__name__)

trained_model = pickle.load(open("model6.pkl","rb"))

@app.route('/')
def home():
    # return jsonify({"Hello World":"1"})
    ax = request.args.get("Acc_x")
    ay = request.args.get("Acc_y")
    az = request.args.get("Acc_z")
    gx = request.args.get("Gyr_x")
    gy = request.args.get("Gyr_y")
    gz = request.args.get("Gyr_z")

    input = np.array([[ax,ay,az,gx,gy,gz]])
    #input = np.array([[ax,ay,az,gz,gy,gx]])
    result = trained_model.predict(input).tolist()[0]
    
    temp={0:'Bike',1:'Sit',2:'Stairsdown',3:'Stairsup',4:'Stand',5:'Walk',6:'Unknow Activity'}
    activity = temp.get(result)
    return jsonify({"Activity":str(activity)})

if __name__=='__main__':
    app.run(debug=True)