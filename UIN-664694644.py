import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string 
import re 
from nltk import tokenize 
import nltk 
from nltk import tag 
from nltk.util import ngrams 
from collections import Counter 
from nltk.corpus import stopwords 
import string 
from nltk.tokenize import wordpunct_tokenize as tokenize 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
import math 
 
 
nltk.download('averaged_perceptron_tagger') 
nltk.download('punkt') 
from nltk import word_tokenize,sent_tokenize 
nltk.download("stopwords") 


movie_rev = pd.read_csv('C:/Users/Vineet/Desktop/Adv Text/Session2/movie_reviews.csv', encoding='latin1', header=None)
print(movie_rev[1][0])
X=movie_rev[1]
y=movie_rev[0]


file_list=[]
for i in range(0, len(X)): 
     temp=X[i]
     file_list.append(temp)
    

    
# change to lower case    
for i in range(0, len(file_list)): 
   file_list[i] = file_list[i].lower()     
 
num_doc = len(file_list) 
 
#removing stopwords 
stopwords=set(stopwords.words('english'))  
#for stemming 
ps=PorterStemmer() 
 
for i in range(num_doc): 
    word_list=file_list[i].split() 
    output=[]
    for s in word_list:
        s=re.sub("\d+",'',s)
        output.append(s)
    while '' in output:
        output.remove('')
#Remove Punctuation 
    output = [''.join(c for c in s if c not in string.punctuation) for s in output] 
#Remove Numbers 
    no_digits = [] 
# Iterate through the string, adding non-numbers to the no_digits list 
    for j in output: 
        if not j.isdigit(): 
            no_digits.append(j)
        
    stemmed=[] 
    for w in no_digits: 
        stemmed.append(ps.stem(w)) 
    for r in stemmed: 
        if not r in stopwords: 
            temp= [w for w in stemmed if not w in stopwords] 
    file_list[i]=' '.join([s for s in temp])
word_tokenized_documents = [word_tokenize (d) for d in file_list] # tokenized docs  
 #create document token set
doc_tokens_set = [] 
doc_tokens_set.append([[key[0] for key in Counter(ngrams(word_tokenized_documents[i], 1, False) ).keys()]  
                       for i in range(num_doc)]) 
doc_tokens_set.append([[key for key in Counter(ngrams(word_tokenized_documents[i], 2, False) ).keys()] 
                       for i in range(num_doc)]) 
 
tokenized_documents = [] 
for i in range(num_doc): 
    tokenized_documents.append(doc_tokens_set[0][i] + doc_tokens_set[1][i]) 
#create all token set 
all_tokens_set = [] 
for i in range(num_doc): 
    all_tokens_set = all_tokens_set + tokenized_documents[i] 
all_tokens_set = set(all_tokens_set)     
 
#idf values 
idf_values = {} 
for tkn in all_tokens_set: 
        contains_token = map(lambda doc: tkn in doc, tokenized_documents) 
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token))) 
#Unigram and Bigram tokens for tf
dlist_1=[]        
dlist_2=[]        
for i in range(num_doc): 
    unigram= Counter(ngrams(word_tokenized_documents[i], 1, False)) 
    bigram = Counter(ngrams(word_tokenized_documents[i], 2, False)) 
    dlist_1.append(unigram) 
    dlist_2.append(bigram) 
 #Create tf matrix
final_tf = [] 
for i in range(num_doc): 
    total=sum(dlist_1[i].values()) + sum(dlist_2[i].values()) 
     
    temp_1 = dict(dlist_1[i]) 
    for key, val in temp_1.items(): 
        temp_1[key] = val/total 
         
    temp_2 = dict(dlist_2[i]) 
    for key, val in temp_2.items(): 
        temp_2[key] = val/total 
     
    temp_1.update(temp_2) 
    final_tf.append(temp_1) 
     
for i in range(num_doc): 
    for key, val in final_tf[i].items(): 
        if len(key) == 1: 
            final_tf[i][key]  = idf_values[key[0]]*val 
        else: 
            final_tf[i][key]  = idf_values[key]*val 

    tf_idf = [] 
    for key in all_tokens_set: 
        temp_val = [] 
        for i in range(num_doc):
            if not isinstance(key, tuple): 
                key = tuple([key]) 
                 
            if key in final_tf[i].keys(): 
                temp_val.append(final_tf[i][key]) 
            else: 
                temp_val.append(0) 
        tf_idf.append(temp_val)
        tf_idf_vf=list(map(list, zip(*tf_idf)))
#Features extracted and stored in X_train
X_train = np.asarray(tf_idf_vf)
y_train = np.array(y)

D = X_train.shape[1] #Number of features
K = max(y_train)+1 #Number of classes assuming class index starts from 0
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))
num_examples = X_train.shape[0]

step_size = 0.05
reg = 0.1 # regularization strength

def mse(predict, label):
    return np.mean(np.square(predict - label))
    
MSE = []
cross_loss = []
for i in range(200):
  
    # evaluate class scores, [N x K]
    scores = np.dot(X_train, W) + b 
  
    # compute the class probabilities using sigmoid function as activation function

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples), y_train])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print ("iteration %d: loss %f" % (i, loss))
  
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y_train] -= 1
    dscores /= num_examples
  
    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X_train.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
  
    dW += reg*W # regularization gradient
  
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    
    print (corect_logprobs)
    cross_loss.append(loss)
    
    scores = np.dot(X_train, W) + b
    predicted_class = np.argmax(scores, axis=1)
    
    MSE.append(mse(np.array(predicted_class), y_train))

    
# Checking for final predicted class 
scores = np.dot(X_train, W) + b
#All the classes are getting predicting correctly
predicted_class = np.argmax(scores, axis=1)
#Training error is 0
print ('Train Error : %0.2f' % mse(np.array(predicted_class), y_train))

#Plotting the cross entropy loss and mean square error
import matplotlib.pyplot as plt
plt.plot(cross_loss)
plt.figure()
plt.plot(MSE)