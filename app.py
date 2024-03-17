from flask import Flask,request,jsonify
import numpy as np
import pickle

app=Flask(__name__)

trained_model = pickle.load(open("model4.pkl","rb"))

@app.route('/')
def home():
    # return jsonify({"Hello World":"1"})
    ax = float(request.args.get("Acc_x"))
    ay = float(request.args.get("Acc_y"))
    az = float(request.args.get("Acc_z"))
    gx = float(request.args.get("Gyr_x"))
    gy = float(request.args.get("Gyr_y"))
    gz = float(request.args.get("Gyr_z"))
    grx = float(request.args.get("Gra_x"))
    gry = float(request.args.get("Gra_y"))
    grz = float(request.args.get("Gra_z"))

    input = np.array([[ax,ay,az,gx,gy,gz,grx,gry,grz]])
    #input = np.array([[ax,ay,az,gz,gy,gx]])
    result = trained_model.predict(input).tolist()[0]
    
    temp={0:'LAYING',1:'SITTING',2:'STANDING',3:'WALKING',4:'WALKING_DOWNSTAIRS',5:'WALKING_UPSTAIRS'}
    activity = temp.get(result)
    return jsonify({"Activity":str(activity)})

if __name__=='__main__':
    app.run(debug=True)