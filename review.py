#I have two models for training purpose SVM and Naive Bayes model

#importing libraries here
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading data from TSV file
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


import nltk
#ltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re
corpus = []

for i in range(0, 1000):
    #keeping only the alphabets 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    #converting uppercase to lowercase
    review = review.lower()
    
    #spliting all the words
    review = review.split()
    
    #removing words like 'the', 'that', etc and performing stemming operation eg playing->play
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #joining the splited words
    review = ' '.join(review) 
    
    #corpus contains all the reviews
    corpus.append(review)


#li = [word for word in stopwords.words('english')]
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)

#maping all the words to all the reviews and storing them in a sparse matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values;


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#training the model with Naive Bayes machine learning model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#printing accuracy
print(classifier.score(X_test, y_test))


#training the model with Support Vector machine learning model
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
                                                                       
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#printing accuracy

print(classifier.score(X_test, y_test))


