# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:33:35 2021

@author: krish
"""


import flask
from flask import Flask, jsonify, request
import json
import numpy as np
import pickle
import flasgger
from flasgger import Swagger
from data_input import data_in

best_model = pickle.load(open('./models/best_model.sav', 'rb'))

app = Flask(__name__)
Swagger(app)

@app.route('/')
def home():
    return('welcome!!')

@app.route('/predict', methods=["GET"])
def predict():
    
    """Let's predict you salary
    ---
    parameters:
        - name: rating
          in: query
          type: number
          required: true
        - name: net_experience
          in: query
          type: number
          required: true
        - name: jr
          in: query
          type: number
          required: true
        - name: senior
          in: query
          type: number
          required: true
        - name: bachelor
          in: query
          type: number
          required: true
        - name: masters
          in: query
          type: number
          required: true
        - name: posting_frequency
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
            
    """
    rating = request.args.get('rating')
    net_experience = request.args.get('net_experience')
    jr = request.args.get('jr')
    senior = request.args.get('senior')
    bachelor = request.args.get('bachelor')
    masters = request.args.get('masters')
    posting_frequency = request.args.get('posting_frequency')
    x_in = [rating, net_experience, jr, senior, bachelor, masters, posting_frequency]
    x_in = np.array(x_in).reshape(1,-1)

    pred = best_model.predict(x_in)
    
    prediction = round(np.exp(pred), 2)
    
    return "Your predicted salary is : " + str(prediction)


if __name__ == '__main__':
 app.run(debug=True)
