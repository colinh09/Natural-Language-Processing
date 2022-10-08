# Sentiment Analysis of Game Reviews From GameStop
# By Colin Hwwang

import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# there were a lot of warnings that didn't really mean much that took up a lot of space
import warnings
warnings.filterwarnings("ignore")

# read file from excel sheet containing the data give reviews a numeric index
data = pd.read_csv('gamestop_reviews.csv')
data = data.sample(frac=1).reset_index(drop=True)

# we only want the actual review and the score given to the game
data = data[['review_description', 'rating']]

# visualize the data by putting the number of reviews for each rating in a bar graph
# it is evident that there are significantly more 5's than other ratings
x = [1, 2, 3, 4, 5]
ratings = data['rating'].value_counts()
y = [ratings[1], ratings[2], ratings[3], ratings[4], ratings[5]]
fig = plt.figure(figsize = (10, 5))
plt.bar(x, y)
plt.show()

# applying a regular expression to the review descriptiosn to getting rid of non letters/numbers
data['review_description'] = data['review_description'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
# tokenize the words, keep only the 5000-1 most frequent words, convert to lowercase
tokenizer = Tokenizer(num_words=5000, lower=True, split=" ")
# update the interal vocab based off the game reviews
tokenizer.fit_on_texts(data['review_description'].values)
# this will transform the text of each review into a sequence of integers
X = tokenizer.texts_to_sequences(data['review_description'].values)
# want all game review sequences to have the same length
X = pad_sequences(X)
# using the sequential model

model = keras.Sequential()
# ---- Architecture 1 ----
# model.add(keras.layers.Embedding(5000, 32, input_length=X.shape[1]))
# model.add(keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))
# model.add(keras.layers.SimpleRNN(32, dropout=0.4, recurrent_dropout=0.2))
# model.add(keras.layers.Dense(5, activation='softmax'))

# ---- Architecture 2 ----
model.add(keras.layers.Embedding(5000, 32, input_length=X.shape[1]))
model.add(keras.layers.LSTM(32, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))
model.add(keras.layers.LSTM(32, dropout=0.4, recurrent_dropout=0.2))
model.add(keras.layers.Dense(5, activation='softmax'))

# ---- Architecture 3 ----
# model.add(keras.layers.Embedding(5000, 64, input_length=X.shape[1]))
# model.add(keras.layers.GRU(32, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))
# model.add(keras.layers.GRU(32, dropout=0.4, recurrent_dropout=0.2))
# model.add(keras.layers.Dense(5, activation='softmax'))

# define loss function, optimizer, and get accuracy from each epoch
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# this use one hot vectors for ratings
# 5 will be 10000, 4 will be 01000, and so on
y = pd.get_dummies(data['rating']).values

# split the data using a 20:80 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# define # of epoch and the batch size
batch_size = 32
epochs = 20

# trains model. Verbose to display info about each epoch
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# saving the model to potentally use it more later
model.save('GR-LSTM-20epochs.h5')

# predict ratings using trained model
predictions = model.predict(X_test)

# should be a better way to organize this data?
# works just fine though even if messy
one_count, two_count, three_count, four_count, five_count = 0, 0, 0, 0, 0
realone, realtwo, realthree, realfour, realfive = 0, 0, 0, 0, 0
for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==4:
        five_count += 1
    elif np.argmax(prediction)==3:
        four_count += 1
    elif np.argmax(prediction)==2:
        three_count += 1
    elif np.argmax(prediction)==1:
        two_count += 1
    else:
        one_count += 1
    if np.argmax(y_test[i])==4:
        realfive += 1
    elif np.argmax(y_test[i])==3:    
        realfour += 1
    elif np.argmax(y_test[i])==2:    
        realthree += 1
    elif np.argmax(y_test[i])==1:    
        realtwo += 1
    else:
        realone +=1

# arrange data into a bar graph to evaluate results
index = np.arange(5)
bar_width = 0.35
fig, ax = plt.subplots()
y1 = [one_count, two_count, three_count, four_count, five_count]
y2 = [realone, realtwo, realthree, realfour, realfive]
realRating = ax.bar(index, y2, bar_width, label = "Real Ratings")
predRating = ax.bar(index + bar_width, y1, bar_width, label = "Predicted Ratings")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['1', '2', '3', '4','5'])
ax.legend()
plt.show()