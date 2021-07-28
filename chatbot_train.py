import nltk
from nltk import data
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
nltk.download('punkt')
# nltk.download('')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

############### Creating Varibles ###############

words = []
classes = []
documents = []
ignore_words = ['?', '!']

############### Loading Intents ###############

data_file = open('intents.json').read()
intents = json.loads(data_file)

############### Tokenizing Sentences in Patterns ###############

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

############### Lemmatizing Words ###############

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

############### Printing size of Documents, words and classes ###############

print(len(documents), " Documents")
print(len(classes), ' Classes')
print(len(words), ' Words')

############### Saving Words and Classes as pickle file ###############

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

############### Training and Testing Data ###############

# Create Training data
training = []

# Create an empty array for the output
output_empty = [0] * len(classes)

# 