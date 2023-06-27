import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv("C:/Users/bcbot/DS Projects/news.csv")

print(df.shape)
print(df.head())

# Get labels

labels = df.label
print(labels.head())

''' 
Here I create the test and train data, the input (x) variable is the text of the article (df['text']) and the output (y) variable is whether the new is fake or real (labels).
'test_size' is set to 0.2, meaning 20% of the data will be set aside as the test data and the remaining 80% will be the data the model is trained on.
'random_state' is similar to seeting a random seed and just ensures reproducibility of the model on the same data.
'''

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

'''
Here I initialize a TfidfVectorizer object. 'stop_words' is a pre-set list of common words that will be removed without being weighted. 
Of note, there seems to be some debate over whether usign this feature is beneficial or not, worth additional review.
'max_df' sets the document frequency cap at which a word is considered important. 
Essentially this says that if a word appears in 70% or more of the documents then it should be considerd a 'common' word and not considered important to the text analysis
'''

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

'''
Here I run the vectorizer object on the input variable for both the train and test set to determine the importance of each word both 
in relation to the individual document and the overall document corpus. The output is a data set with a tuple index of (document #, index of word in document) and it's TF-IDF value.
The TF-IDF value is the term frequency (how many times the word appears in that document) times the IDF which is 
a number that scales down as the number of documents containing that word increases. 
Therefore, a high TF-IDF indicates that the word is used a lot in that specific document, but isn't very common and therefore has a high importance to the content of that document specifically.

Of note, the fit_transform is used on the train data set to standardize the data (x-mean/sd calculation). 
The transform method is used on the test data set so that it is standardizeed using the same mean and SD as the train set rather than calculating the mean/SD of the test set.
This is done to avoid biasing the model by giving it additional information about the test set that would influence the prediction rate.
'''

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


'''
Next I create a PassiveAgressiveClassifier object. This is a online machine learning algorithim that is often used for large sources of data.
A good example of when you would use this data is if I wanted to use this fake news detector on a data source like twitter.
The model is considered passive agressive because it is 'passive' every time the model is correct (i.e. no change to model) and 'agressive' when the model is wrong.

The model uses the TF-IDF values of the words in the documents to assess the probability of that document being Real or Fake and classify it accordingly.
'''

pac = PassiveAggressiveClassifier(max_iter=50, random_state=145)
pac.fit(tfidf_train, y_train)

'''
Once the model is trained it is tested on the test set of data and a score is calculated. For this model I achieved a 92.66% accuracy rate
'''

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {round(score*100, 2)}%')

'''
Finally I review the precision of the results to see my type 1 and type 2 error rate.
'''

c_mat = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

c_mat_d = ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels= ['FAKE', 'REAL'])
c_mat_d.plot()
plt.show()