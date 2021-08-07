import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier

# reading the penguins data set
penguins = pd.read_csv('D:/achu/programs/Projects/Streamlit/penguins_cleaned.csv')

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)


# separating x and y
X = df.drop('species', axis=1)
y = df['species']

# random forest classifier
clf = RandomForestClassifier()
clf.fit(X, y)

pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
