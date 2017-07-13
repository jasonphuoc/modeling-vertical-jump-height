from sklearn import svm
from sklearn.metrics import r2_score
import pandas as pd

#use first 30 samples for training/validation

df = pd.read_csv("/Users/jasonla/Desktop/UCLA/Health Analytics/vertical-jump-prediction/test-data/vertical-jump-height-nfl.csv")
df1 = df.head(30)
df2 = df.tail(13)

#choose predictors

predictors = ['Height (in)', 'Weight (lbs)']
outcome = ['Vert Leap (in)']
model = svm.SVR()
model.fit(df1[predictors], df1[outcome])

'''
#get r2 score using last 13 samples 
target = []
prediction = []
for index, row in df2.iterrows():
    predictors = []
    predictors.append(row['Height (in)'])
    predictors.append(row['Weight (lbs)'])
    prediction.append(model.predict(predictors))
    target.append(row['Vert Leap (in)'])

print r2_score(target, prediction)
'''

print model.predict(df1[predictors])
print r2_score(df1[outcome], model.predict(df1[predictors]))