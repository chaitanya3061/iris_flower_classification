from flask import Flask, render_template, request
import numpy as np
app = Flask(__name__)
import pickle

@app.route('/')

def abc():
    return render_template("index.html")  

def load_model():
    global model
    with open('savedmodel.sav', 'rb') as f:
        model = pickle.load(f)
@app.route('/predict',methods=['POST'])
def basic():
    if request.method == 'POST':
        sepal_length = request.form['sepallength']
        sepal_width = request.form['petalwidth']
        petal_length = request.form['petallength']
        petal_width = request.form['petalwidth']
        sepal_length=float(sepal_length)
        sepal_width=float(sepal_width)
        petal_length=float(petal_length)
        petal_width=float(petal_width)
        data = np.array([sepal_length,sepal_width,petal_length,petal_width ])[np.newaxis, :]
        prediction = model.predict(data) 
        a=str(prediction[0])
        if a=='Iris-setosa' :
            return render_template('index1.html', setosa='setosa')
        elif a=='Iris-virginica':
            return render_template('index1.html', virginica='virginica')
        else:
            return render_template('index1.html',versicolor='versicolor') 
    return render_template('index.html')
        
        
if __name__ == '__main__':
    load_model()
    app.run(debug=True)