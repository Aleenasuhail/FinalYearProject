import bs4 as bs
import urllib.request
import re

import nltk
import numpy as np
import random
import string
import warnings
warnings.filterwarnings('ignore')


raw_data = urllib.request.urlopen('https://www.dmu.ac.uk/current-students/new-starters/faqs.aspx')
raw_data = raw_data.read()

html_data = bs.BeautifulSoup(raw_data,'lxml')

all_paragraphs =html_data.find_all('p')

article_content = ""

for p in all_paragraphs:
    article_content += p.text

article_content =  article_content.lower()# converts to lowercase

article_content  = re.sub(r'\[[0-9]*\]', ' ', article_content )
article_content = re.sub(r'\s+', ' ', article_content )

sentence_list = nltk.sent_tokenize(article_content)
article_words= nltk.word_tokenize(article_content )

nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = nltk.stem.WordNetLemmatizer()

def LemmatizeWords(words):
    return [lemmatizer.lemmatize(word) for word in words]

remove_punctuation= dict((ord(punctuation), None) for punctuation in string.punctuation)

def RemovePunctuations(text):
    return LemmatizeWords(nltk.word_tokenize(text.lower().translate(remove_punctuation)))

greeting_input_texts = ("hey", "heys", "hello", "morning", "evening","greetings",)
greeting_replie_texts = ["hey", "hey hows you?", "*nods*", "hello there", "ello", "Welcome, how are you"]

def reply_greeting(text):

    for word in text.split():
        if word.lower() in greeting_input_texts:
            return random.choice(greeting_replie_texts)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def give_reply(user_input):
    chatbot_response=''
    sentence_list.append(user_input)
    word_vectors = TfidfVectorizer(tokenizer=RemovePunctuations, stop_words='english')
    vecrorized_words = word_vectors.fit_transform(sentence_list)
    similarity_values = cosine_similarity(vecrorized_words[-1], vecrorized_words)
    similar_sentence_number =similarity_values.argsort()[0][-2]
    similar_vectors = similarity_values.flatten()
    similar_vectors.sort()
    matched_vector = similar_vectors[-2]
    if(matched_vector ==0):
        chatbot_response=chatbot_response+"I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response = chatbot_response +sentence_list[similar_sentence_number]
        return chatbot_response

flag=True
print("Hello, I am DeMon, I am your virtual assistant:")
while(flag==True):
    user_input = input()
    user_input = user_input .lower()
    if(user_input !='bye'):
        if(user_input =='thanks' or user_input =='thank you very much'  or user_input =='thank you'):
            flag=False
            print("DeMon: Most welcome")
        else:
            if(reply_greeting(user_input)!=None):
                print("DeMon: "+reply_greeting(user_input))
            else:
                print("DeMon: ",end="")
                print(give_reply(user_input))
                sentence_list.remove(user_input)
    else:
        flag=False
        print("DeMon: Take care, bye ..")


 
