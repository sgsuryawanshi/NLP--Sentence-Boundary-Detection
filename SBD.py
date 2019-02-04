#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import re,sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
def readdata(train,test):
with open(train) as f:
traindata = f.read().splitlines()
with open(test) as g:
testdata = g.read().splitlines()
trainraw=[]
testraw=[]
for n in range(0,len(traindata)):
trainraw+=[re.findall(r'\S+',traindata[n])]
for n in range(0,len(testdata)):
testraw+=[re.findall(r'\S+',testdata[n])]
# genearating variables for train data
regex=re.compile(r'\w+\.')
dataset=[]
dataset_test=[]
test_data=[]
train_data=[]
for i in range(0,len(trainraw)):
if trainraw[i][2]!="TOK":
if regex.findall(trainraw[i][1]) !=[]:
if i != len(trainraw)-1:
dataset+=[trainraw[i]+[trainraw[i][1][:-1]]+[trainraw[i+1][1]]]
else:
dataset+=[trainraw[i]+[trainraw[i][1][:-1]]+[" "]]
for i in range(0,len(dataset)):
if len(dataset[i][3])>3:
dataset[i]=dataset[i]+[1]
else:
dataset[i]=dataset[i]+[0]
for i in range(0,len(dataset)):
if dataset[i][3][0].isupper()==True:
dataset[i]=dataset[i]+[1]
else:
dataset[i]=dataset[i]+[0]
for i in range(0,len(dataset)):
if dataset[i][4][0].isupper()==True:
dataset[i]=dataset[i]+[1]
else:
dataset[i]=dataset[i]+[0]
for i in range(0,len(dataset)):
train_data+=[dataset[i][2:]]
# additional features
# no of vowels in left word, no of vowels in right word,length of left word
vowels=["A","E","I","O","U"]
for i in range(0,len(train_data)):
train_data[i][1].upper() in vowels
Lcounter = 0
Rcounter=0
for letter in train_data[i][1].upper():
if letter in vowels:
Lcounter += 1
for letter in train_data[i][2].upper():
if letter in vowels:
Rcounter += 1
train_data[i]+=[Lcounter]
train_data[i]+=[Rcounter]
train_data[i]+=[len(train_data[i][1])]
# genearating variables for testdata
for i in range(0,len(testraw)):
if testraw[i][2]!="TOK":
if regex.findall(testraw[i][1]) !=[]:
if i != len(testraw)-1:
dataset_test+=[testraw[i]+[testraw[i][1][:-1]]+[testraw[i+1][1]]]
else:
dataset_test+=[testraw[i]+[testraw[i][1][:-1]]+[" "]]
for i in range(0,len(dataset_test)):
if len(dataset_test[i][3])>3:
dataset_test[i]=dataset_test[i]+[1]
else:
dataset_test[i]=dataset_test[i]+[0]
for i in range(0,len(dataset_test)):
if dataset_test[i][3][0].isupper()==True:
dataset_test[i]=dataset_test[i]+[1]
else:
dataset_test[i]=dataset_test[i]+[0]
for i in range(0,len(dataset_test)):
if dataset_test[i][4][0].isupper()==True:
dataset_test[i]=dataset_test[i]+[1]
else:
dataset_test[i]=dataset_test[i]+[0]
for i in range(0,len(dataset_test)):
test_data+=[dataset_test[i][2:]]
# additional features for test data
for i in range(0,len(test_data)):
test_data[i][1].upper() in vowels
Lcounter = 0
Rcounter=0
for letter in test_data[i][1].upper():
if letter in vowels:
Lcounter += 1
for letter in test_data[i][2].upper():
if letter in vowels:
Rcounter += 1
test_data[i]+=[Lcounter]
test_data[i]+=[Rcounter]
test_data[i]+=[len(test_data[i][1])]
# genearating data frame for train and test data for all features
train=pd.DataFrame(train_data,columns=['EOS_NEOS','Left','Right','Left>3','L_is_upper','R_is_upper','L_Vowels','R_Vowels','Len_L'])
test=pd.DataFrame(test_data,columns=['EOS_NEOS','Left','Right','Left>3','L_is_upper','R_is_upper','L_Vowels','R_Vowels','Len_L'])
# replacing left and right word with index to make it compatible with Decision trees
train["L_Word"]=train.index
train["R_Word"]=train.index
test["L_Word"]=test.index
test["R_Word"]=test.index
train_data_Final=train.replace({"EOS":1,"NEOS":0})
test_data_Final=test.replace({"EOS":1,"NEOS":0})
# Dataset for core features
Y=train_data_Final[["EOS_NEOS"]]
X=train_data_Final[['L_Word','R_Word','Left>3','L_is_upper','R_is_upper']]
test_data_F=test_data_Final[['L_Word','R_Word','Left>3','L_is_upper','R_is_upper']]
test_data_F1=test_data_Final[["EOS_NEOS"]]
X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.2,stratify=Y, random_state=1
)
rf = DecisionTreeClassifier(max_depth=2,random_state=1)
rf.fit(X_train,Y_train)
Y_pred=rf.predict(X_test)
accuracy_score(Y_test,Y_pred)
test_pred=rf.predict(test_data_F)
testaccuracy=accuracy_score(test_data_F1,test_pred)
print("Accuracy for core features:",round(testaccuracy*100,2),"%")
print("Training model for all features...")
Y_all=train_data_Final[["EOS_NEOS"]]
X_all=train_data_Final[['L_Word','R_Word','Left>3','L_is_upper','R_is_upper','L_Vowels','R_Vowels','Len_L']]
test_data_F_all=test_data_Final[['L_Word','R_Word','Left>3','L_is_upper','R_is_upper','L_Vowels','R_Vowels','Len_L']]
test_data_F1_all=test_data_Final[["EOS_NEOS"]]
# Data split
X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(
X_all, Y_all, test_size=0.2,stratify=Y_all, random_state=1
)
rf_all = DecisionTreeClassifier(max_depth=2,random_state=1)
rf_all.fit(X_train_all,Y_train_all)
Y_pred_all=rf_all.predict(X_test_all)
acc=accuracy_score(Y_test_all,Y_pred_all)
#print("Training Accuracy for all feature:",round((acc*100),2),"%")
#print("Predicting for test data...")
test_pred_all=rf_all.predict(test_data_F_all)
testaccuracy_all=accuracy_score(test_data_F1_all,test_pred_all)
print("Accuracy for all features:",round(testaccuracy_all*100,2),"%")
print("Training model for 3 implemented features...")
Y_3=train_data_Final[["EOS_NEOS"]]
X_3=train_data_Final[['L_Vowels','R_Vowels','Len_L']]
test_data_F_3=test_data_Final[['L_Vowels','R_Vowels','Len_L']]
test_data_F1_3=test_data_Final[["EOS_NEOS"]]
# Data split
X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(
X_3, Y_3, test_size=0.2,stratify=Y_3, random_state=1
)
rf_3 = DecisionTreeClassifier(max_depth=2,random_state=1)
rf_3.fit(X_train_3,Y_train_3)
Y_pred_3=rf_3.predict(X_test_3)
acc3=accuracy_score(Y_test_3,Y_pred_3)
#print("Training Accuracy for 3 feature:",round((acc3*100),2),"%")
#print("Predicting for test data...")
test_pred_3=rf_3.predict(test_data_F_3)
testaccuracy_3=accuracy_score(test_data_F1_3,test_pred_3)
print("Accuracy for 3 added features:",round(testaccuracy_3*100,2),"%")
out=pd.DataFrame(test_pred)
out1=test
out1["Pred"]=out
outputfile=out1[["EOS_NEOS","Left","Pred"]]
outputfile["EOS_NEOS"].replace({1:"EOS",0:"NEOS"})
#print(" 1st 5 rows of SBD.test.out")
#print(outputfile.head())
outputfile.to_csv("SBD.test.out")
print("SBD.test.out saved to working directory")
def maincall():
train=sys.argv[1]
test=sys.argv[2]
bi_list=readdata(train,test)
if __name__=='__main__':
maincall()
