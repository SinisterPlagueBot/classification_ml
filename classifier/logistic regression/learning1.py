from sklearn.model_selection import train_test_split
from sklearn import linear_model
from mymodel import accuracy,LogisticRegression
import pandas as pd

dataSet = pd.read_csv('fake_bills.csv', sep=';', na_values=' ')
dataSet['margin_low']=dataSet['margin_low'].fillna(dataSet['margin_low'].mean())
missing_columns = dataSet.columns[dataSet.isnull().any()].tolist()

print("Columns with missing values:", missing_columns)
x_train,x_test,y_train,y_test = train_test_split(dataSet.drop('is_genuine',axis=1),dataSet['is_genuine'],random_state=30,train_size=0.8)
model = LogisticRegression(proba='tanh',random_state=42,max_iter=2000,learning_rate=0.1)
model.fit(x_train,y_train)
print(accuracy(y_train,model.predict(x_train)))
print(accuracy(y_test,model.predict(x_test)))
