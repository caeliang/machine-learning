import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data=pd.read_csv('Breast Cancer Survival Prediction with Machine Learning\BRCA.csv')
print(data.head())

# print(data.isnull().sum())

#bos verileri atar
data.dropna(inplace=True)

print(data.isnull().sum())

#veri hakkında bilgi verir
# data.info()

# print(data.Gender.value_counts())

# stage = data["Tumour_Stage"].value_counts()
# transactions = stage.index
# quantity = stage.values

# figure = px.pie(data, 
#              values=quantity, 
#              names=transactions,hole = 0.5, 
#              title="Tumour Stages of Patients")
# figure.show()

# # ER status
# print(data["ER status"].value_counts())
# # PR status
# print(data["PR status"].value_counts())
# # HER2 status
# print(data["HER2 status"].value_counts())

# surgery = data["Surgery_type"].value_counts()
# transactions = surgery.index
# quantity = surgery.values
# figure = px.pie(data, 
#              values=quantity, 
#              names=transactions,hole = 0.5, 
#              title="Type of Surgery of Patients")
# figure.show()

data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"] = data["PR status"].map({"Positive": 1})
data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
data["Gender"] = data["Gender"].map({"MALE": 0, "FEMALE": 1})
data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4})
print(data.head())

x = np.array(data[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                   'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                   'HER2 status', 'Surgery_type']])
y = np.array(data[['Patient_Status']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

model = SVC()
model.fit(x_train, y_train)
features = np.array([[36.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
print(model.predict(features))
