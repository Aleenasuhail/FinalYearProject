import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNetwork
#opening and loading json data
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] #empty list that will later hold patterns and tags
# loop through each sentence in intents patterns in intents
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w) #xetents the array append will put array in array
        # add to xy pair
        xy.append((w, tag)) #pattern and the corresponding tag

# stem and lower each word
ignore_words = ['?', '.', '!','@']
#ignores words in ignore_words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# create training data
X_train = []  #bag of words
y_train = [] #associated number with each tags
for (pattern_sentence, tag) in xy: #unpacking the tuple xy.append((w, tag))
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words) #bag_of_words takes tokenized sentence and all_words as an arguement
    X_train.append(bag) #empty list X_tain gets updated with bag
    # y: CrossEntropyLoss needs only labels
    label = tags.index(tag) #labels to the tag for eg greetings 0
    y_train.append(label)

X_train = np.array(X_train) #converting to numpy array
y_train = np.array(y_train) #converting to numpy array

# Hyper-parameters
num_epochs = 300 #training loops
batch_size = 8    #no. of samples
learning_rate = 0.001  #updates weights
input_size = len(X_train[0]) #size of input layer
hidden_size = 6 #no. of hidden neurons
output_size = len(tags) #size of output layer

#create new class ChatDataset which inherits dataset
class ChatDataset(Dataset):

    def __init__(self): #implement init function which gets self
        self.n_samples = len(X_train) #strores the length of X_train array
        self.x_data = X_train #strores data from X_train array
        self.y_data = y_train #strores data from Y_train array

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
#creating data loader
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #check if gpu is available

model = NeuralNetwork(input_size, hidden_size, output_size).to(device) #pushing model to device

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer is used to update the attributes such as weight
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #calculates adaptive learning rate for parameters

# Train the model
for epoch in range(num_epochs):    #epochs in sequence e.g 0,1.....50
    for (words, labels) in train_loader: #unpack words and labels
        words = words.to(device) #push to device
        labels = labels.to(device) #push to device

        # Forward pass
        outputs = model(words) #words are passed through the nn

        loss = criterion(outputs, labels)  #calculates loss using predicted output and actaul labels

        # Backward and optimize
        optimizer.zero_grad() #empty gradient
        loss.backward()  #calculate backpropogation
        optimizer.step() #updates parameters based on loss.backward()
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')


#saving data
data = {
"model_state": model.state_dict(), #create dictionary to save model state
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth" #define file name pth for pytorch
torch.save(data, FILE) #serialize it and save to pickeled file

print(f'training complete. file saved to {FILE}')
