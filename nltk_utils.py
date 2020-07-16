import numpy as np
import nltk
# nltk.download('punkt') only once
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import casual_tokenize
stemmer = SnowballStemmer('finnish')
def tokenize(sentence):
    
    #creates array of words by slicing sentences
    
    return nltk.word_tokenize(sentence)


def stem(word):
    
    #stemming  finds the root word  form of the word
    
    return stemmer.stem(word.lower())
# takes word as attribute converts it to lower case 

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    each word found is 1 and 0 for not found

    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)  #example [0,0,0,0,0]
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
