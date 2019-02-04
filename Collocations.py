import nltk
import os
from nltk import FreqDist, bigrams
import math

with open("Collocations") as file_SGS:
    m = file_SGS.read().splitlines()
sgsstr = ''.join(m)

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~`'''

new_str = ""
for char in sgsstr:
   if char not in punctuations:
       news_tr = new_str + char
       len(new_str)

unigrams = nltk.word_tokenize(new_str)
unigrams_count_SGS = len(unigrams)
len(unigrams)

import nltk
nltk.download('punkt')
bigrams = list(nltk.bigrams(unigrams))
len(bigrams)

from nltk.probability import FreqDist
a_unigrams = FreqDist()
for word in unigrams:
    a_unigrams[word]+=1
a_unigrams
a_bigrams = FreqDist()
for word in bigrams:
    a_bigrams[word]+=1
a_bigrams
J = len(bigrams)
j = len(unigrams)
J

a_bigrams.items()

bi_val_SGS=[]
uni_val_SGS = []

for i in a_unigrams.items():
    uni_val_SGS.append(i)

for i in a_bigrams.items():
    bi_val_SGS.append(i)
    len(uni_val_SGS)

chi_list_ss = []

def chi_sq(e,f):
    count = 0
    for i in range(0,len(bi_val_SGS)):
        if bi_val_SGS[i][0][0] == e and bi_val_SGS[i][0][1] == f:
            p = bi_val_SGS[i][1]

    for i in range(0,len(bi_val_SGS)):
        if bi_val_SGS[i][0][1] == f:
            count+= bi_val_SGS[i][1]
        q = count - p
    for i in range(0,len(uni_val_SGS)):
        if uni_val_SGS[i][0] == e:
            x = uni_val_SGS[i][1]
            r = x - p
    s = J - p -q - r
    chi = (((p-(((p+q)*(p+r))/J)) **2)/ (((p+q)*(p+r))/J ))+(((q-(((p+q)*(q+s))/J))**2)/(((p+q)*(q+s))/J))+(((r-(((p+r)*(r+s))/J))**2)/(((p+r)*(r+s))/J))+(((s-(((s+r)*(s+q))/J))**2)/(((s+r)*(q+s))/J))

    val = [e , f ,chi]
    chi_list_ss.append(val)

def chi_square_SGS():
    for i in range(0,len(bi_val_SGS)):
        e = bi_val_SGS[i][0][0]
        f = bi_val_SGS[i][0][1]

        ch = chi_sq(e,f)


    def sort_Second(val):
        return val[2]

    chi_list_ss.sort(key = sort_Second, reverse = True)
    for i in chi_list_ss[:20]:
        print(i)

chi_square_SGS()

pmi_list_ss = []
def pm(e,f):
    for i in range(0,len(bi_val_SGS)):
        if bi_val_SGS[i][0][0] == e and bi_val_SGS[i][0][1] == f:
            p = bi_val_SGS[i][1]
    for i in range(0,len(uni_val_SGS)):
        if uni_val_SGS[i][0] == e:
            q = uni_val_SGS[i][1]
        if uni_val_SGS[i][0] == f:
            r = uni_val_SGS[i][1]
    prob = (p/N)/((q/n)*(r/n))

    pmi = math.log(prob)
        p_val = [e , f , pmi]
    pmi_list_ss.append(p_val)
    def pmi():
    for i in range(0,len(bi_val_SGS)):
        e = bi_val_SGS[i][0][0]
        f = bi_val_SGS[i][0][1]

        pmi = pm(e,f)

    def sort_Second(val):
        return val[2]

    pmi_list_ss.sort(key = sort_Second, reverse = True)
    for i in pmi_list_ss[:20]:
        print(i)


pmi()
