
import random
import json

import torch

from model import NeuralNetwork
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #cgeck if gpu support is available
#opening and loading json data
with open('intents.json', 'r') as json_data: #opens intents.json in read mode
    intents = json.load(json_data)

#opens saved file from train.py
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"] #get input_size
hidden_size = data["hidden_size"] #get hidden_size
#hidden_size is no of neurons in the hidden layer
output_size = data["output_size"] #get output_size
all_words = data['all_words']     #get all_words
tags = data['tags']                #get tags
model_state = data["model_state"]   #save model state

model = NeuralNetwork(input_size, hidden_size, output_size).to(device) #
model.load_state_dict(model_state) #gets learnt parameters
model.eval() #set model to evaluation mode

#implement actual chat
robot_name = "DeMon"
print("Hi I am DeMon Welcome to De Montfort University I am here to help! (type 'quit' to exit)")
while True:

    sentence = input("You: ")
    if sentence == "quit": #if sentence is quit then terminate the chat
        break

    sentence = tokenize(sentence) #split the sentence
    X = bag_of_words(sentence, all_words) #takes tokenized sentence as input and all_wprds from saved data.pth file
    X = X.reshape(1, X.shape[0]) #shape in form of rows no columns
    X = torch.from_numpy(X).to(device) #converting to torch tensor
#using model get preticted output
    output = model(X)
    _, predicted = torch.max(output, dim=1)
#get actual tags
    tag = tags[predicted.item()]
    #softmax activation function is used to get the probabilty distribution in order to decide whether the output should be printed to user or not
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.82:

        for intent in intents['intents']:
            if tag == intent["tag"]: #preticted tag = actual tag
                print(f"{robot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{robot_name}: I am sorry I do not understand...")
