############### Imports ###############

import nltk
from nltk import data
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
###############
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

# training set --> bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenize words for patterns
    pattern_words = doc[0]
    # lemmatize each word - Create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '0' for each tag and '1' for current tag(for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle Our Features and Turn into Array
random.shuffle(training)
training = np.array(training)

# Create Train and Test lists : X - patterns, y - intents
x_train = list(training[:,0])
y_train = list(training[:,1])
print("--Training Data Created--")