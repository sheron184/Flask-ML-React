from flask import Flask
from flask import request
from flask_mysqldb import MySQL
import joblib
import pandas 
import numpy 
import joblib
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import json
 
app = Flask(__name__) 
  
doctors = {
    "0": "Dr. Nethuka",
    "1": "Dr. Dasun",
    "2": "Dr. Nimal",
    "3": "Dr. Chamara",
    "4": "Dr. Malith",
    "5": "Dr. Kasun",
    "6": "Dr. Isuru",
}


# MYSQL CONFIG 
#app.config['MYSQL_HOST'] = 'localhost' 
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'hackathon'
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

@app.route('/profile')
def my_profile():
    response_body = {
        "name": "Nagato",
        "about" :"Hello! I'm a full stack developer that loves python and javascript"
    }

    return response_body

@app.route("/login", methods=['POST'])
def login():
    mysql = MySQL(app)
    
    if (request.method == 'POST'):
        username = request.form.get('username')
        password = request.form.get('password')
        print(username)
        if(mysql):

            cursor = mysql.connection.cursor()
            
            # cursor.execute(' ' 'INSERT INTO users VALUES(%s,%s)' ' ', (username, password))
            
            # mysql.connection.commit()
            cursor.close()
    
    return 'ok'

@app.route("/predict")
def predict():
    
    loaded_model = joblib.load('./CNN/text_classify.pkl')
    loaded_vectorizer = joblib.load('./CNN/vectorizer.pkl')
    
    predict_text = request.args.get('text')

    print(f"================== {predict_text} =================")
    text = loaded_vectorizer.transform([predict_text])
    text  = text.toarray()
    text.shape
    
    text_predict = loaded_model.predict(text)
    text_predict
    to_arr = text_predict.tolist()
    payload = json.dumps(to_arr)
    
    return payload
    
