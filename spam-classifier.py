import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#get data 
data = pd.read_csv("data/sample-message.csv")


X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("training messages:")
print(X_train)

print("testing mesages:")
print(X_test)

print("training labels:")
print(y_train)

print("testing labels:")
print(y_test)