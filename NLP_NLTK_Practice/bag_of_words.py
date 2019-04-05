# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:52:22 2019

@author: vdokku
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

for data in text_data:
    print("This is the text :>", data)

# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# what is the difference between the FIT and FIT_TRANSFORM.

# Show feature matrix
bag_of_words.toarray()


feature_names = count.get_feature_names()

feature_names

# So we are creating the dataframe, with the column names. 
pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

from nltk.stem.porter import PorterStemmer

"""

Stemming reduces a word to it's stem .
A word will be reduced to a stem.

It makes the text more comparable across observations. 




Stemming reduces a word to its stem by identifying and 
removing affixes (e.g. gerunds) while keeping the root 
meaning of the word. NLTK’s PorterStemmer 
implements the widely used Porter stemming algorithm.


"""

tokenized_words = ['i', 'am', 'humbled', 'by', 
                   'this', 'traditional', 'meeting']

"""
Stem Words

Stemming reduces a word to its stem by identifying and 
removing affixes (e.g. gerunds) while keeping the root 
meaning of the word. NLTK’s PorterStemmer implements the 
widely used Porter stemming algorithm.


"""





# Create stemmer
porter = PorterStemmer()

# This is like an enhanced for loop. 
# Apply stemmer
[porter.stem(word) for word in tokenized_words]

# Create text
text_data = ['   Interrobang. By Aishwarya Henriette     ',
             'Parking And Going. By Karl Gautier',
             '    Today Is The night. By Jarek Prakash   ']


stop_white_space = [input_string.strip() for input_string in text_data]


stop_white_space


# Load libraries
from nltk import pos_tag
from nltk import word_tokenize


text_data = "Chris loved outdoor running"

# tag part of the speech.

text_tagged = pos_tag(word_tokenize(text_data))

text_tagged

"""
Tag 	Part Of Speech
NNP 	Proper noun, singular
NN 	    Noun, singular or mass
RB 	    Adverb
VBD 	Verb, past tense
VBG 	Verb, gerund or present participle
JJ 	    Adjective
PRP 	Personal pronoun

"""



"""

is a measure of originality of a word by comparing the number of word
appears in a doc with the number of docs the word appears in .


TF - IDF 
Term frequency Inverse document frequence 

"""

# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])



tfidf = TfidfVectorizer()
output_tfidf = tfidf.fit_transform(text_data)

output_tfidf.toarray()

tfidf.get_feature_names()


# Create data frame
pd.DataFrame(output_tfidf.toarray(), columns=tfidf.get_feature_names())

"""

     beats     best     both    brazil  germany       is      love   sweden
0  0.00000  0.00000  0.00000  0.894427  0.00000  0.00000  0.447214  0.00000
1  0.00000  0.57735  0.00000  0.000000  0.00000  0.57735  0.000000  0.57735
2  0.57735  0.00000  0.57735  0.000000  0.57735  0.00000  0.000000  0.00000

"""


from nltk.tokenize import word_tokenize, sent_tokenize

string = "The science of today is the technology of tomorrow. Tomorrow is today."

tokensized_word_1= word_tokenize(string)

tokensized_word_1

tokenized_sentenses = sent_tokenize(string)
tokenized_sentenses





