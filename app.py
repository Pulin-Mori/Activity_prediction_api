from flask import Flask,request,jsonify
import numpy as np
import pickle

app=Flask(__name__)

trained_model = pickle.load(open("model2.pkl","rb"))

@app.route('/')
def home():
    # return jsonify({"Hello World":"1"})
    ax = request.args.get("Acc_x")
    ay = request.args.get("Acc_y")
    az = request.args.get("Acc_z")
    gx = request.args.get("Gyr_x")
    gy = request.args.get("Gyr_y")
    gz = request.args.get("Gyr_z")
    #grx = request.args.get("Gra_x")
    #gry = request.args.get("Gra_y")
    #grz = request.args.get("Gra_z")

    #input = np.array([[ax,ay,az,grx,gry,grz,gx,gy,gz]])
    input = np.array([[ax,ay,az,gz,gy,gx]])
    result = trained_model.predict(input).tolist()[0]
    
    temp={0:'LAYING',1:'SITTING',2:'STANDING',3:'WALKING',4:'WALKING_DOWNSTAIRS',5:'WALKING_UPSTAIRS'}
    activity = temp.get(result)
    return jsonify({"Activity":str(activity)})

if __name__=='__main__':
    app.run(debug=True)