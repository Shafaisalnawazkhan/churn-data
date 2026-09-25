from flask import Flask, render_template, request
import pandas as pd 
import joblib
app=Flask(__name__)
model=joblib.load("churn_model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    customer={
        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Tenure": int(request.form["Tenure"]),
        "Usage Frequency": int(request.form["Usage Frequency"]),
        "Support Calls": int(request.form["Support Calls"]),
        "Payment Delay": int(request.form["Payment Delay"]),
        "Subscription Type": request.form["Subscription Type"],
        "Contract Length": (request.form["Contract Length"]),
        "Total Spend": float(request.form["Total Spend"]),
        "Last Interaction": int(request.form["Last Interaction"])
    }
    
    input_data=pd.DataFrame([customer])
    prediction=model.predict(input_data)[0]
    if prediction==1:
        result="The customer is likely to CHURN." 
    else:
        result="The customer is likely to STAY."  
        
    return render_template("index.html", prediction_text=result)

if __name__=="__main__":
    app.run(debug=True)    