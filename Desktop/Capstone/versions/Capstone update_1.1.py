#!/usr/bin/env python
# coding: utf-8

# In[72]:


#warnings :)
import warnings
warnings.filterwarnings('ignore')
#from tkinter import *
import nltk
nltk.download('punkt')


# In[73]:


# Top level window
from tkinter import * 
import tkinter as tk
frame = tk.Tk()
frame.title("TextBox Input")
frame.geometry('1000x1000')

# Function for getting Input from textbox and printing it at label widget
 
def printInput():
    global inp
    inp = inputtxt.get(1.0, "end-1c")
    #print(f"Your current input is :{inp}")
    lbl.config(text = "Provided Input: "+inp)

#Textbox creation
inputtxt = tk.Text(frame,height = 20, width = 40,bg = "light blue")

inputtxt.pack()

# Button Creation
printButton = tk.Button(frame,text = "Enter",command = printInput)
printButton.pack()
#print(inp)
# sleep(5)
#Convert to lower case the user input
print("===============")
print(inp)
print("===============")

import string
raw_docs = inp.lower()
#print(raw_docs)

#Remove punctuation

my_string = raw_docs
output = my_string.translate(str.maketrans('', '', string.punctuation))
#print(f"the output after removing punctuations: {output}")

#Removing whitespace

no_ws_doc = output.strip()
#print(no_ws_doc)

#Remove Numbers
import re
no_num = re.sub(r'\d+','',no_ws_doc)
print(no_num)

#Word tokenize

# from nltk.tokenize import word_tokenize
# tokenized_docs = word_tokenize(no_ws_doc)
# print(f"The output for word tokenization : {tokenized_docs}")

# #Remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
post_doc = [word for word in no_num.split()if not word in stopwords.words()]
print (f"The output after removing stopwords:{post_doc}")

# Stemming and Lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#porter = PorterStemmer()
# wordnet = WordNetLemmatizer()

# preprocessed_docs = []

# for doc in post_doc:
#     final_doc = []
#     for word in doc:
#         #final_doc.append(porter.stem(word))
#         final_doc.append(wordnet.lemmatize(word))
    
#     preprocessed_docs.append(final_doc)

#print( preprocessed_docs)

lbl = tk.Label(frame, text = "Enter the sentence to compare", font='bold')
lbl.pack()

lab_msg = tk.Label(frame , text = "The Sentence from the Database is below:", font='bold')
lab_msg.pack()
lbl1 = tk.Label(frame, text =no_ws_text )
lbl1.pack()
labl2 = tk.Label(frame, text= name_text)
labl2.pack()
frame.mainloop()


# In[31]:


pip install pymongo


# In[32]:


#MongoDB
from pymongo import MongoClient
import dns

client = MongoClient("mongodb+srv://user:test@cluster0.3o3d3.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

db = client.get_database('sentences')
records=db['random_sentence']

#count document without providing any filter 
#records.count_documents({})
total= db.random_sentence.count_documents({})
print(total)


# In[33]:


import random
cur=db.random_sentence.find({},{"_id":0})
x= list(cur)
rand_sentence= random.choice(x)
print(rand_sentence)
text = str(rand_sentence)
word = 'sentence'
text = text.replace(word, "")
print(text)


# In[42]:


#lower the mongo sentence
text_lower= text.lower()
print(f" Coverted to lower text: {text_lower}")

#punctuation removed
text_punc = text_lower.translate(str.maketrans('', '', string.punctuation))

print(f" Punctuation removed : {text_punc }")


#Removing whitespace

no_ws_text = text_punc.strip()

print(f"Whitespace is removed :{no_ws_text}")

#Remove Numbers
import re
no_num_text = re.sub(r'\d+','',no_ws_text)
print(f"Numbers are removed :{no_num_text}")


#Named Entity Recognition

from nltk import word_tokenize,pos_tag,ne_chunk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
name_text= ne_chunk(pos_tag(word_tokenize(no_num_text)))
print(f"Named entities are listed as : {name_text}")

#Remove stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stoplist = stopwords.words('english')
postpa = [word for word in no_num_text.split()if word not in stoplist]
print(f"Stop words are removed : {postpa}")

#synonym
# from nltk.corpus import wordnet

# synonyms=[]
# for i in postpa:
#     for syn in wordnet.synsets(i):
#         for lemma in syn.lemma():
#             synonyms.append(lemma.name())
# print(synonyms)


#Stamming and Lemminization 

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
porter = PorterStemmer()
lemma = WordNetLemmatizer()

lemmatized_string = ' '.join([lemma.lemmatize(words) for words in postpa])
print(lemmatized_string)  

# print(lemma.lemmatize(postpa))
stem_string = ' '.join([porter.stem(words)for words in postpa])
print(stem_string)


#Word tokenize
# from nltk.tokenize import word_tokenize
# tokenized_docs_text = word_tokenize(lemmatized_string)
# print(f"The output for word tokenization : {tokenized_docs_text}")


# In[58]:


from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data

# # Load a text file if required
# text = "Pete ate a large cake. Sam has a big mouth."
output = ""

# Load the pretrained neural net
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Tokenize the text
#tokenized = tokenizer.tokenize(text)

# Get the list of words from the entire text
words = postpa

# Identify the parts of speech
tagged = nltk.pos_tag(words)

for i in range(0,len(words)):
    replacements = []

    # Only replace nouns with nouns, vowels with vowels etc.
    for syn in wordnet.synsets(words[i]):

        # Do not attempt to replace proper nouns or determiners
        if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
            break
        
        # The tokenizer returns strings like NNP, VBP etc
        # but the wordnet synonyms has tags like .n.
        # So we extract the first character from NNP ie n
        # then we check if the dictionary word has a .n. or not 
        word_type = tagged[i][1][0].lower()
        if syn.name().find("."+word_type+"."):
            # extract the word only
            r = syn.name()[0:syn.name().find(".")]
            replacements.append(r)

    if len(replacements) > 0:
        # Choose a random replacement
        replacement = replacements[randint(0,len(replacements)-1)]
        output = output + " " + replacement
    else:
        # If no replacement could be found, then just use the
        # original word
        output = output + " " + words[i]

print(output)

#Stamming and Lemminization 

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
porter = PorterStemmer()
lemma = WordNetLemmatizer()

lemmatized_string = ''.join([lemma.lemmatize(words) for words in output])
print(lemmatized_string)  

# print(lemma.lemmatize(postpa))
stem_string = ''.join([porter.stem(words)for words in output])
print(stem_string)


complete_text= output+ "."+ stem_string+"."+lemmatized_string
print(complete_text)

# from nltk.tokenize.treebank import TreebankWordDetokenizer
# TreebankWordDetokenizer().detokenize([stem_string])
# TreebankWordDetokenizer().detokenize([lemmatized_string])


# In[75]:


from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# string1 = "My name is cat"
# string2 = "i am a cat"
similar(complete_text, post_doc)

