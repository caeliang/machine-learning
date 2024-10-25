import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

data=pd.read_csv('Covid-19 Deaths Prediction (Case Study)\COVID19 data for overall INDIA.csv')
print(data.tail())

print(data.isnull().sum())

data.drop(['Date'],axis=1,inplace=True)
print(data.head())

# import plotly.express as px
# fig = px.bar(data, x='Date_YMD', y='Daily Confirmed')
# fig.show()

# cases = data["Daily Confirmed"].sum()
# deceased = data["Daily Deceased"].sum()

# labels = ["Confirmed", "Deceased"]
# values = [cases, deceased]

# fig = px.pie(data, values=values, 
#              names=labels, 
#              title='Daily Confirmed Cases vs Daily Deaths', hole=0.5)
# fig.show()

# fig = px.bar(data, x='Date_YMD', y='Daily Deceased')
# fig.show()
from autots import AutoTS
model = AutoTS(forecast_length=1, frequency='infer', ensemble='simple')
model = model.fit(data, date_col="Date_YMD", value_col='Daily Deceased', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)