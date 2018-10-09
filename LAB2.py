import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets #Iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib

iris = datasets.load_iris()

data_X = iris.data
data_X = preprocessing.scale(data_X)
data_y = iris.target
'''
plt.scatter(data_X[:,0],data_X[:,1],c = data_y)
plt.show()
'''
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
'''
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(X_train, y_train)
# evaluating GNB
print ("accuracy on testing dataset: %s" % model.score(X_test, y_test))
'''

'''
from sklearn.svm import SVC

scaler = preprocessing.StandardScaler() ## It is usually a good idea to scale the data for SVM training.
data_X = scaler.fit_transform(data_X)

#2.1
clf_1 = SVC(kernel="poly", degree=4)

clf_1.fit(X_train, y_train)
print ("accuracy on testing dataset: %s" % clf_1.score(X_test, y_test))

#2.2
clf_2 = SVC(kernel="rbf")
clf_2.fit(X_train, y_train)
print ("accuracy on testing dataset: %s" % clf_2.score(X_test, y_test))

#2.3
#SVR
from sklearn.svm import SVR
# evaluation
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
##param_grid = dict(gamma=gamma_range, C=C_range)
#model upgrade, try to modify the value of C & gamma
for i in [1,10,1e2,1e3,1e4]:
    rbfSVR_score=[]
    for j in np.linspace(0.1,1,10):
        rbf_svr = SVR(kernel = 'rbf', C=i, gamma=j)
        rbf_svr_predict = cross_val_predict(rbf_svr, data_X, data_y, cv=5)
        rbf_svr_score = cross_val_score(rbf_svr, data_X, data_y, cv=5)
        rbf_svr_meanscore = rbf_svr_score.mean()
        rbfSVR_score.append(rbf_svr_meanscore)
    plt.plot(np.linspace(0.1,1,10),rbfSVR_score,label='C='+str(i))
    plt.legend()
plt.xlabel('gamma')
plt.ylabel('score')
plt.show()


#3
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

file_name = input("Enter the file name\n")
with open(file_name,'r') as f1:

    ess = f1.read()
    stokens = nltk.sent_tokenize(ess)
    wtokens = nltk.word_tokenize(ess)
    
    for w in wtokens:
        print(lemmatizer.lemmatize(w)) #lemmatize
    
    bigrams = nltk.ngrams(wtokens,2) #bigrams
    bigramDist = nltk.FreqDist(bigrams) #frequency test
    print(bigramDist)
    print(bigramDist.most_common(5))

    sen_list = []
    for i in range(5):
        words_1 = bigramDist.most_common(5)[i][0][0]
        words_2 = bigramDist.most_common(5)[i][0][1]
        for s in stokens:
            if ((words_1 in s) and (words_2 in s) and (s not in sen_list)):
                sen_list.append(s)
    print(sen_list)

'''

#4
#KNN
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

#model k range (1-50)
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn_model = knn.fit(X_train, y_train)
    print('k-NN accuracy:{}' .format(knn_model.score(X_test,y_test)))
    
    plt.scatter(k,knn_model.score(X_test,y_test),s=30,c='red',marker='o',alpha=0.5)

plt.show()


