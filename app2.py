
#app2 run.py


from flask import Flask , request , jsonify , render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
updated_scaler = pickle.load(open("scaler.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")
    
@app.route('/static/<path:path>')
def serve_static(path):
    return app.send_static_file(path)


    

@app.route('/predict',methods=["POST"])
def predict():
    user_features = [int(x) for x in request.form.values()]
    print("User Features:", user_features) 
    
    if not user_features:
        return "No data received from the form."
    
    prefinal = [np.array(user_features)]
    scaled_values = updated_scaler.transform(prefinal)

    predictions = model.predict(scaled_values)

    print("predicted:", predictions[0])
    if(predictions[0] == 1):
        output = "true"
    elif (predictions[0] == 0):
        output = "false"
    else:
        output = "maybe or may not be"
    
    return render_template("index.html", display="output = {}".format(output))



if __name__ == "__main__":
    app.run(debug=True)