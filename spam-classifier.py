import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#get data 
data = pd.read_csv("data/sample-message.csv")


X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("training messages:")
print(X_train)

print("testing mesages:")
print(X_test)

print("training labels:")
print(y_train)

print("testing labels:")
print(y_test)

print("Real answers:")
print(y_test.values)

accuracy = accuracy_score(y_test, y_pred)

print("accuracy:")
print(accuracy)