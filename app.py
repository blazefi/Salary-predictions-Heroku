from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
import pandas as pd
import os
from utils.nlp_utils import Word2VecVectorizer
from utils.data_preprocessing import Preprocess
from gensim.models import KeyedVectors

app = Flask(__name__)

#join the chunks

def join(source_dir, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')
     
    # Get a list of the file parts
    parts = ['word2vec-part_1','word2vec-part_2','word2vec-part_3', 'word2vec-part_4', 'word2vec-part_5', 'word2vec-part_6', 'word2vec-part_7', 'word2vec-part_8', 'word2vec-part_9', 'word2vec-part_10', 'word2vec-part_11', 'word2vec-part_12', 'word2vec-part_13', 'word2vec-part_14', 'word2vec-part_15', 'word2vec-part_16', 'word2vec-part_17', 'word2vec-part_18', 'word2vec-part_19', 'word2vec-part_20', 'word2vec-part_21', 'word2vec-part_22', 'word2vec-part_23', 'word2vec-part_24', 'word2vec-part_25']
 
    # Go through each portion one by one
    for file in parts:
         
        # Assemble the full path to the file
        path = source_dir+file
         
        # Open the part
        input_file = open(path, 'rb')
         
        while True:
            # Read all bytes of the part
            bytes = input_file.read(read_size)
             
            # Break out of loop if we are at end of file
            if not bytes:
                break
                 
            # Write the bytes to the output file
            output_file.write(bytes)
             
        # Close the input file
        input_file.close()
         
    # Close the output file
    output_file.close()
join(source_dir='utils/word2vec-chunks/', dest_file="utils/word2vec.bin", read_size = 20000000)



#load glove embeddings
filename = 'utils/word2vec_model.bin'

model = KeyedVectors.load_word2vec_format(filename, binary=True)

def glove_embedded(X, col,train_data):
  vectorizer = Word2VecVectorizer(model)
  X_embed = vectorizer.fit_transform(X[col].apply(str))
  train_data = np.concatenate((X_embed, train_data), axis=1)
  
  return train_data


#load model
xgb_model = pickle.load(open('models/xgb_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():   
     
    x_in  = list(request.form.values())
    
    columns = ['Job_position', 'Company', 'Location', 'requirements', 'rating', 'experience', 'posting_frequency']

    input_df = pd.DataFrame(columns = columns)

    for j in range(len(x_in)):
        input_df.loc[0, columns[j]] = x_in[j]
 
    input_df = Preprocess()(input_df)

    train_data = input_df.select_dtypes(exclude='object').values
    
    for col in input_df.select_dtypes(include='object').columns:
        train_data = glove_embedded(input_df, col, train_data)

    pred = xgb_model.predict(train_data)
    prediction = np.round(np.exp(pred), 2)
   
    return render_template('index.html', prediction_text='Your predicted annual salary is {}'.format(prediction))

if __name__ == '__main__':
 app.run(debug=True)
