#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
#Loading the model pickle file 
input_pickle = open("Simran_NuthalapatiYesukumar_model.pkl","rb")
random_forest_model = pickle.load(input_pickle)

@app.route('/')
def home():
    return render_template('index.html')
#Prediction of results 
@app.route('/predict',methods=["POST"])
def predict():
    input_feat = [x for x in request.form.values()]
    final_model_feat = [np.array(input_feat)]
    preds = random.predict(final_model_feat)
    return render_template('index.html', preds_txt = 'The Species to which the fish belongs to is {}'.format(str(preds)))

if __name__=='__main__':
    app.run()


# In[ ]:




