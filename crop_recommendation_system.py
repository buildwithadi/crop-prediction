import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

crop = pd.read_csv("D:\Aditya\Research Paper\Research-project\Crop_recommendation.csv")

ncrop = crop
ncrop = ncrop.drop(columns='label')

crop_dict = {
            'rice':1,
            'maize':2,
            'chickpea':3,
            'kidneybeans':4,
            'pigeonpeas':5,
            'mothbeans':6,
            'mungbean':7,
            'blackgram':8,
            'lentil':9,
            'pomegranate':10,
            'banana':11,
            'mango':12,
            'grapes':13,
            'watermelon':14,
            'muskmelon':15,
            'apple':16,
            'orange':17,
            'papaya':18,
            'coconut':19,
            'cotton':20,
            'jute':21,
            'coffee':22
            }

crop['label'] = crop['label'].map(crop_dict)

X = crop.drop('label', axis = 1)
y = crop['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Normalization: we are normalizing the data between (0,1)
mx = MinMaxScaler()
mx.fit(X_train)
X_train = mx.transform(X_train)
X_test = mx.transform(X_test)

#Standardization: we are converting the mean: 0 and standard deviation: 1 of the given data.
sc = StandardScaler()

sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#Training Model
randclf = RandomForestClassifier()
randclf.fit(X_train, y_train)
y_pred = randclf.predict(X_test)

print(accuracy_score(y_test,y_pred))

dict_crop = {
            1:'rice',
            2:'maize',
            3:'chickpea',
            4:'kidneybeans',
            5:'pigeonpeas',
            6:'mothbeans',
            7:'mungbean',
            8:'blackgram',
            9:'lentil',
            10:'pomegranate',
            11:'banana',
            12:'mango',
            13:'grapes',
            14:'watermelon',
            15:'muskmelon',
            16:'apple',
            17:'orange',
            18:'papaya',
            19:'coconut',
            20:'cotton',
            21:'jute',
            22:'coffee'
            }

def recommendation(N,P,K,temperature,humidity,ph,rainfall):
  features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
  mx_features = mx.transform(features)
  sc_mx_features = sc.transform(mx_features)
  prediction = randclf.predict(sc_mx_features)
  return f'You should grow {(dict_crop[prediction[0]]).capitalize()} on your soil.'


# n = int(input("N: "))
# p = int(input("p: "))
# k = int(input("k: "))
# temp = int(input("temp: "))
# humid = int(input("humidity: "))
# ph = int(input("ph: "))
# rain = int(input("rainfal: "))
# print(recommendation(n,p,k,temp,humid,ph,rain))


# Web Framework ----------------------------------------------------------------------------------------------

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    N = round(float(request.form['N']))
    P = round(float(request.form['P']))
    K = round(float(request.form['K']))
    temp = round(float(request.form['temp']))
    humid = round(float(request.form['humid']))
    ph = round(float(request.form['ph']))
    rain = round(float(request.form['rain']))
    return recommendation(N,P,K,temp,humid,ph,rain)
    

if __name__ == '__main__':
    app.run(debug=True)


"""
Maize: 95 38 22 19 61 5 100 
Banana: 107 71 55 29.42 83.96 6 117.22 
MothBean: 3 49 18 27.91 64.70 3.69 32.67
Lentil: 19 79 19 20 67.76 6.67 42.89
Grapes: 24 130 195 30 81.54 6.11 67.125
"""

