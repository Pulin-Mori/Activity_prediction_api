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

    if any(v is None for v in [ax, ay, az, gx, gy, gz]):
        return jsonify({"error": "One or more required parameters are missing."}), 400

    try:
        # Convert string values to float
        ax = float(ax)
        ay = float(ay)
        az = float(az)
        gx = float(gx)
        gy = float(gy)
        gz = float(gz)

        input = np.array([[ax,ay,az,gx,gy,gz]])
        result = trained_model.predict(input).tolist()[0]
    
        temp={0:'Bike',1:'Sit',2:'Stairsdown',3:'Stairsup',4:'Stand',5:'Walk',6:'Unknow Activity'}
        activity = temp.get(result)
        return jsonify({"Activity":str(activity)})
    except ValueError:
        return jsonify({"error": "One or more parameters are not valid floating-point numbers."}), 400

if __name__=='__main__':
    app.run(debug=True)