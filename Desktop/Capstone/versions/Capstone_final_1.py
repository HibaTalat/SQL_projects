#!/usr/bin/env python
# coding: utf-8

# In[2]:


#warnings :)
import warnings
warnings.filterwarnings('ignore')
from tkinter import *
import nltk
nltk.download('punkt')


# # 1. Mongo DB Setup 
# 
# Pulling data from MongoAtlas 
# https://cloud.mongodb.com/v2/6121b9107a12b36bfc551739#metrics/replicaSet/612313cda2e2d25be365b868/explorer/sentences/random_sentence/find
# 
# 

# In[4]:


pip install pymongo


# In[16]:


get_ipython().system('pip install textblob')


# In[7]:


#MongoDB
from pymongo import MongoClient
import dns

client = MongoClient("mongodb+srv://user:test@cluster0.3o3d3.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

db = client.get_database('sentences')
records=db['random_sentence']

#count document without providing any filter records.count_documents({})
total= db.random_sentence.count_documents({})
print("==================")
print(total)
print("==================")


# In[11]:


import random
cur=db.random_sentence.find({},{"_id":0})
x= list(cur)
rand_sentence= random.choice(x)
print("==================")
print(rand_sentence)
print("==================")

text = str(rand_sentence)
word = 'sentence'
text = text.replace(word, "")
#printing a random sentence from the 15 sentences we have in Mongo DB


print(text)
print("==================")


# In[15]:


import string
#lower the mongo sentence
text_lower= text.lower()
print(f" Coverted to lower text: {text_lower}")
print("==================")

#punctuation removed
text_punc = text_lower.translate(str.maketrans('', '', string.punctuation))

print(f" Punctuation removed : {text_punc }")
print("==================")


#Removing whitespace

no_ws_text = text_punc.strip()

print(f"Whitespace is removed :{no_ws_text}")
print("==================")

#Remove Numbers
import re
no_num_text = re.sub(r'\d+','',no_ws_text)
print(f"Numbers are removed :{no_num_text}")
print("==================")


#Named Entity Recognition

from nltk import word_tokenize,pos_tag,ne_chunk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
name_text= ne_chunk(pos_tag(word_tokenize(no_num_text)))
print(f"Named entities are listed as : {name_text}")
print("==================")

#Remove stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stoplist = stopwords.words('english')
postpa = [word for word in no_num_text.split()if word not in stoplist]
print(f"Stop words are removed : {postpa}")
print("==================")

#Replacing with synonyms
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data

# # Load a text file if required
# text = "Pete ate a large cake. Sam has a big mouth."
output = ""



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

print(f'the output is : {output}')
print("==================")

#Stamming and Lemminization 

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
porter = PorterStemmer()
lemma = WordNetLemmatizer()

lemmatized_string = ''.join([lemma.lemmatize(words) for words in output])
print(f'The lematized version of the sentence is : {lemmatized_string}')  
print("==================")

# print(lemma.lemmatize(postpa))
stem_string = ''.join([porter.stem(words)for words in output])
print(f'The stemmized version of the sentence is : {stem_string}')

complete_text= output+ "."+ stem_string+"."+lemmatized_string
print(f'The complete sentence including synonym , lemmatized and stemmized is : {complete_text}')
print("==================")

# from nltk.tokenize.treebank import TreebankWordDetokenizer
# TreebankWordDetokenizer().detokenize([stem_string])
# TreebankWordDetokenizer().detokenize([lemmatized_string])
print("=====================\n")
from nltk.tokenize import word_tokenize
tokenized_docs_text = word_tokenize(complete_text)
print(f"The output for word tokenization : {tokenized_docs_text}")
print("==================")

#Spellcheck using textblob

from textblob import TextBlob
spell_check_text= TextBlob(complete_text)

print(f'The spell check has been performed on the sentence:{spell_check_text.correct()}')
print("==================")


# # 2. Creating GUI for user to add a sentence
# Using Tkinter python library 

# In[ ]:


# Top level window

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

#Button Creation
printButton = tk.Button(frame,text = "Enter",command = printInput)
printButton.pack()
#print(inp)
# sleep(5)

#Convert to lower case the user input
print("===============")
print(f' You have provided this sentence:{inp}')
print("===============")

import string
raw_docs = inp.lower()
print(f'Converted the sentence to lower case:{raw_docs}')

#Remove punctuation

my_string = raw_docs
output_withno_punc = my_string.translate(str.maketrans('', '', string.punctuation))
print(f"the output after removing punctuations: {output_withno_punc}")
print("===============")

#Removing whitespace

no_ws_doc = output_withno_punc.strip()
print(f'Remove the whitespaces in between sentence:{no_ws_doc}')
print("===============")

#Remove Numbers
import re
no_num = re.sub(r'\d+','',no_ws_doc)
print("===============")

print(f' Removing the numbers:{no_num}')

#Word tokenize

from nltk.tokenize import word_tokenize
tokenized_docs = word_tokenize(no_ws_doc)
print(f"The output for word tokenization : {tokenized_docs}")
print("===============")

# #Remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
post_doc = [word for word in no_num.split()if not word in stopwords.words()]
print("===============")
print (f"The output after removing stopwords:{post_doc}")

print("===============")


lbl = tk.Label(frame, text = "Enter the sentence to compare", font='bold')
lbl.pack()
lab_msg = tk.Label(frame , text = "The Sentence from the Database is below:", font='bold')
lab_msg.pack()
lbl1 = tk.Label(frame, text =no_ws_text )
lbl1.pack()
# labl2 = tk.Label(frame, text= name_text)
# labl2.pack()
frame.mainloop()



# In[45]:


from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
porter = PorterStemmer()
lemma = WordNetLemmatizer()

# # Load a text file if required
# text = "Pete ate a large cake. Sam has a big mouth."
output_in = ""



# Get the list of words from the entire text
words = post_doc

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
        output_in = output_in + " " + replacement
    else:
        # If no replacement could be found, then just use the
        # original word
        output_in = output_in + " " + words[i]

print(f'The output after replacing with the synonyms is : {output_in}')

#Stamming and Lemminization 

lemmatized_string1 = ''.join([lemma.lemmatize(words) for words in output_in])
print(f'The lematized version of the sentence is : {lemmatized_string1}')  

# # print(lemma.lemmatize(postpa))
stem_string1 = ''.join([porter.stem(words)for words in output_in])
print(f'The stemmized version of the sentence is : {stem_string1}')


complete_text1= output_in+ "."+ stem_string1+"."+lemmatized_string1
print(f'The complete sentence including synonym , lemmatized and stemmized is : {complete_text1}')


#Spellcheck using textblob

from textblob import TextBlob
spell_check_text1= TextBlob(complete_text1)
print(f"Spell check:{spell_check_text1.correct()}")

# # from nltk.tokenize.treebank import TreebankWordDetokenizer
# # TreebankWordDetokenizer().detokenize([stem_string])
# # TreebankWordDetokenizer().detokenize([lemmatized_string])
# print("=====================\n")
# from nltk.tokenize import word_tokenize
# tokenized_docs_text = word_tokenize(complete_text)
# print(f"The output for word tokenization : {tokenized_docs_text}")



# # Comparing to sentences and returning the percentage

# In[46]:



import difflib

output_final = str(int(difflib.SequenceMatcher(None, complete_text1, complete_text).ratio()*100))
print(f' The percent of your text match is :{output_final}')

