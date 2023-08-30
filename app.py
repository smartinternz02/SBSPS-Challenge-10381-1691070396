from flask import Flask, render_template, request
#from Admission_New import calc

import pandas as pd



app = Flask(__name__)# interface between my server and my application wsgi

import pickle
model = pickle.load(open(r'C:\Users\nagap\Downloads\IBM\model.pkl','rb'))


@app.route('/')#binds to an url
def helloworld():
    return render_template("index.html")

@app.route("/prediction", methods=["GET"])
def redirect_internal():
    return render_template("/prediction.html", code=302)

@app.route('/predicted', methods =['POST'])#binds to an url
def login():
    p =request.form["gs"]
    q= request.form["ts"]
    if (q=="1"):
        q_val=1
    if (q=="0"):
        q_val=0
    r= request.form["ur"]
    if (r=="1"):
        r_val=1
    if (r=="2"):
        r_val=2
    if (r=="3"):
        r_val=3
    if (r=="4"):
        r_val=4
    if (r=="5"):
        r_val=5
    if (r=="0"):
        r_val=0
    s= request.form["sop"]
    t= request.form["lor"]
    #u= request.form["host"]
    #if (u=="1"):
     #   u_val=1
    #if (u=="0"):
      #  u_val=0
    v= request.form["rnd"]
    if (v=="1"):
        v_val=1
    if (v=="0"):
        v_val=0
    
    import requests

    # NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
    API_KEY = "rWSI2SsOJ_rEIroauoUl_A6yfso5iC3seqUoSXahE_w7"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": ["Age","Gender","Stream" ,"Internships","CGPA","HistoryOfBacklogs"], "values": [[float(p),float(q_val),float(r_val),float(s),float(t),float(v_val)]]}]}

    response_scoring = requests.post('https://eu-gb.ml.cloud.ibm.com/ml/v4/deployments/91b5b687-7fa2-4747-93f8-77d78bd4aa87/predictions?version=2021-05-01', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    #print(response_scoring.json())
    op = response_scoring.json()
    #print(op['predictions'])
    prediction_value = op['predictions'][0]['values'][0][0]
    confidence_value = op['predictions'][0]['values'][0][1][1]
    #print(prediction_value)







    
    #return render_template("prediction.html",submit_to_check_result = "The predicted chance is  " + str(prediction_value)+" percentage" )
    
    #output = model.predict([calc(int(p),int(q_val),int(r_val),int(s),int(t),int(u_val),int(v_val))])
    #print(output)  
    
    #new_data = pd.DataFrame({
    #'Age': [int(p)],
    #'Gender': [int(q_val)],
    #'Stream': [int(r_val)],
    #'Internships': [int(s)],
    #'CGPA': [int(t)],
    #'HistoryOfBacklogs': [int(v_val)]
#})
    #output = model.predict(new_data)
    #print(output)
    if(prediction_value==1):
        return render_template("prediction.html",submit_to_check_result = "Congratulations!! you have high probability to get placed. Chance is = " + str((confidence_value*100))+"%")
    else:
        return render_template("prediction.html",submit_to_check_result = "Sorry!! you still need to improve.  Chance is = " + str((confidence_value*100))+ "%")
    #return render_template("prediction.html",submit_to_check_result = "The predicted chance is  " + str((output[0]*100))+" percentage" )
    #return render_template("prediction.html",submit_to_check_result = "The predicted chance is  " + str((output[0]*100))+" percentage" )

#@app.route('/admin')#binds to an url
#def admin():
   # return "Hey Admin How are you?"

if __name__ == '__main__' :
    app.run(debug= False)
    