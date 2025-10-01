"""Script to train and save disease prediction models."""
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

train = pd.read_csv('data/training.csv')
cols = train.columns
x = train[cols[:-1]]
y = train['prognosis']

rfc = RandomForestClassifier()
rfc.fit(x, y)
svc = SVC()
svc.fit(x, y)

# Create the model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the models
joblib.dump(rfc, 'model/rfc.model')
joblib.dump(svc, 'model/svc.model')
