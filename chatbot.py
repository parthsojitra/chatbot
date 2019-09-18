import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
from translate import Translator
import speech_recognition as sr
import pyttsx3
import random
import json
import pickle
import datetime
import calendar
import wikipedia
import sys

with open("first.json") as f:
	data = json.load(f)

lemm = WordNetLemmatizer()

#print(data["intents"])
with open("chatbot_data.pickle", "rb") as f:
	words, labels, training, output = pickle.load(f)

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])

net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.load("chatbot_model.tflearn")


def bag_of_words(s, words):
	bag = [0 for i in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [lemm.lemmatize(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)


r=sr.Recognizer()
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[6].id)
g = Translator(to_lang="Gujarati")
e = Translator(from_lang="Gujarati", to_lang="English")

def speak(audio):
	print('Computer: ' + audio)
	engine.say(audio)
	engine.runAndWait()

def greetMe():
    currentH = int(datetime.datetime.now().hour)
    if currentH >= 0 and currentH < 12:
        speak('Good Morning!')

    if currentH >= 12 and currentH < 18:
        speak('Good Afternoon!')

    if currentH >= 18 and currentH !=0:
        speak('Good Evening!')

def myCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 3
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        print('User: ' + query + '\n')
        #query = e.translate(query)
        #print('User: ' + query + '\n')

    except sr.UnknownValueError:
        speak('Sorry sir! I didn\'t get that! Try typing the command!')
        query = str(input('Command: '))

    return query


def chat():
	greetMe()
	speak('Hello Sir, I am your digital assistant Smiley!')

	warn = True

	while True:
		if int(datetime.datetime.now().hour)>0 and int(datetime.datetime.now().hour)<3 and warn:
			speak("It's too late Sir, you have to take rest")
			warn = False
			continue

		inp = input("You : ")
		inp = inp.lower()

		res = model.predict([bag_of_words(inp, words)])[0]
		res_index = np.argmax(res)
		tag = labels[res_index]
		#print(res)

		if res[res_index] > 0.75:
			if tag == "goodbye":
				r = ["Sad to see you go,Sir", "Talk to you later,Sir", "Goodbye,Sir"]
				speak(random.choice(r))
				break

			elif tag == "datetime":
				d = datetime.date.today()
				dy = d.weekday()
				t = datetime.datetime.now()
				speak("Date : "+str(d))
				speak("Day : "+str(calendar.day_name[dy]))
				speak("Time : "+str(t.strftime('%I:%M:%S %p')))
				continue

			elif tag == "date":
				d = datetime.date.today()
				speak(str(d))
				continue

			elif tag == "day":
				dy = d.weekday()
				speak(str(calendar.day_name[dy]))
				continue

			elif tag == "time":
				t = datetime.datetime.now()
				speak(str(t.strftime('%I:%M:%S %p')))
				continue

			else:
				for tg in data["intents"]:
					if tg["tag"] == tag:
						responses = tg["responses"]
				speak(random.choice(responses))
		else:
			try:
				if inp.startswith("who is") or inp.startswith("who are"):
					query = inp[7:]
					r = wikipedia.summary(query, sentences=3)
					speak(r)

				elif inp.startswith("what is") or inp.startswith("what are"):
					query = inp[8:]
					r = wikipedia.summary(query, sentences=3)
					speak(r)

				else:
					query = inp
					r = wikipedia.summary(query, sentences=3)
					speak(r)

			except:
				speak("Sorry,I can't understand this.")

chat()