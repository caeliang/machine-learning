import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

data=pd.read_csv('Future Sales//advertising.csv') 
print(data.isnull().sum())
X = np.array(data[['TV', 'Radio', 'Newspaper']])
y =np.array (data['Sales'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVR()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(mae)



