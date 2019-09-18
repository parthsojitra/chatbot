import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import datetime
import wikipedia

with open("first.json") as f:
	data = json.load(f)

lemm = WordNetLemmatizer()

words = []
labels = []
docs_pat = []
docs_tag = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_pat.append(wrds)
		docs_tag.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

words = [lemm.lemmatize(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for i in range(len(labels))]

for x, doc in enumerate(docs_pat):
	bag = []

	wrds = [lemm.lemmatize(w.lower()) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	out_raw = out_empty[:]
	out_raw[labels.index(docs_tag[x])] = 1

	training.append(bag)
	output.append(out_raw)

training = np.array(training)
output = np.array(output)

with open("chatbot_data.pickle", "wb") as f:
	pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])

net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("chatbot_model.tflearn")