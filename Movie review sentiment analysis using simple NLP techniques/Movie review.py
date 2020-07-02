
# importing libraries
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# importing dataset
data = pd.read_csv("D:/dataforpython/User_movie_review.csv")

# data to be process
data_review = data["text"]

# check if there are any missing records
data.isnull().sum()

# creating object porterstemmer
stemmer = PorterStemmer()


doc = []
for i in range(0,data.shape[0]):
    text = data_review[i]
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = nltk.word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word not in stopwords.words("english")]
    text = " ".join(text)
    print(text)
    doc.append(text)

# For wordcloud we need to convert list into string because list wont work with wordcloud
doc1 = "".join(doc)

# Wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

wordcloud = WordCloud(width=800,relative_scaling=1.0, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(doc1)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()              # After exceting this we will get wordcloud of most frequent words used in doc1 (for frequent words we will keep relative scaling =1)


# importing countvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=20000)
x = cv.fit_transform(doc).toarray()

# Shape
x.shape

# y = data.iloc[:,0].values    <------ by doing this we get string values so we need to convert it into integer

# We will convert y variable to integer
y = data["class"].map({"Pos":1,"Neg":0})

# Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# After data preprocessing we will look how our data looks like
dataframe = pd.DataFrame(x_train, columns=cv.get_feature_names())
dataframe.head()

# fitting vaive bayes to training (GAUSSIAN)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

# predicting test set results
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)









































