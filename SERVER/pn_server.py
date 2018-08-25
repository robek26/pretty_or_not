from http.server import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import cgi
import sys
from pprint import pprint
import urllib.parse
import codecs
import random
import os
import json
from socketserver import ThreadingMixIn
import threading
from keras.layers import Input, Dense,Dropout
from keras.models import Model
import json
import operator
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from keras import backend as K
import face_recognition as fr

encode_arr = np.load('encodes/100_training_img_encodes.npy')
encode_arr_test = np.load('encodes/287_test_img_encodes.npy')

PORT_NUMBER = 22000

#This class will handles any incoming request from
#the browser 


class myHandler(BaseHTTPRequestHandler):
    
    """
    Custom functions to process incoming and outgoing signals
    
    """
    
    def build_model(self,X_train,y_train):
    
        inputs = Input(shape=(128,))
        inner = Dense(50, activation='tanh')(inputs)
        inner = Dropout(0.7)(inner)
        pred = Dense(1, activation='sigmoid')(inner)
        self.model = Model(inputs=inputs, outputs=pred)
        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        self.model._make_predict_function()
        self.model._make_test_function()
        self.model._make_train_function()
        self.model.fit(X_train, y_train , epochs = 100)
   

    def train_and_test(self,answers):
        global encode_arr
        global encode_arr_test

        answers = answers.split(',')
        answers = np.array(answers)
        X = encode_arr
        y = answers
        X_train = X[:90,:]
        y_train = y[:90,].reshape((90,1))
        X_test = X[90:,:]
        y_test = y[90:,].reshape((10,1))

        with tf.Session(graph=tf.Graph()) as sess:
            
            K.set_session(sess)

            self.build_model(X_train,y_train)

            preds1 = self.model.evaluate(X_test, y_test)
            print ("Loss = " + str(preds1[0]))
            print ("Test Accuracy = " + str(preds1[1]))

            #predict test data

            preds_test1 = self.model.predict(encode_arr_test)


            # divide them to positive results and negative results

            results_pos = {}
            results_neg = {}
            ccnter  = 0
            for i in preds_test1:
                if i[0] < 0.35:
                    results_neg[ccnter] = str((1 - i[0]) * 100) + '%'
                else:
                    results_pos[ccnter] = str(i[0] * 100) + '%'
                ccnter += 1

            # sorting the pos and neg results in descending order

            oo_pos = OrderedDict(sorted(results_pos.items(), key=lambda x: x[1],reverse=True))
            oo_neg = OrderedDict(sorted(results_neg.items(), key=lambda x: x[1],reverse=True))

            oo_total_result = {'positive' : oo_pos , 'negative' : oo_neg}
            return json.dumps(oo_total_result)




    """
    End of Custom
    
    """
    
    #Handler for the GET requests
    def do_GET(self):
        print('do GET')
        #pprint (vars(self))
        self.send_response(200)
        print(self.path)
        if self.path == '/favicon.ico':
            return
         # Send the html message
        if not os.path.isdir(self.path[1:]):
            with open(self.path[1:], 'rb') as reader:
                content = reader.read()
        else:
            con = os.listdir(self.path[1:])
            len_con = len(con)
            if con[0] == '.DS_Store':
                len_con -= 1
            content = str(len_con).encode()
            
        
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        self.wfile.write(content)
       
        return

    #Handler for the POST requests
    def do_POST(self):
        pprint (vars(self))
        length = int(self.headers['Content-Length'])
       
        
        
        
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST'}
        )
       
        
        answers = form['answers'].value
        print(answers)
        
        output = self.train_and_test(answers)
        
        # Send back response after the data is processed
        
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type','text/html')
        
        self.end_headers()
        # Send the html message
        self.wfile.write(output.encode())
        
        
        
        return
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

try:
    #Create a web server and define the handler to manage the
    #incoming request
    server = ThreadedHTTPServer(('', PORT_NUMBER), myHandler)
    print('Started httpserver on port ' , PORT_NUMBER)
    
    #Wait forever for incoming htto requxests
    server.serve_forever()

except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    server.socket.close()