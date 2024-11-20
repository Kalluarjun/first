import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score
import seaborn as sns
df = fetch_california_housing()
#print(df) 
dataset= pd.DataFrame(df.data)
dataset.columns= df.feature_names 
#independent features and dependent features
X= dataset # or X= df.data
y= df.target
#train test split
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train)
#standardization the dataset

scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)

#cross validation

regression= LinearRegression()
regression.fit(X_train, y_train)
mse= cross_val_score(regression, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
#print(np.mean(mse))

#prediction
reg_pred= regression.predict(X_test)
#print(reg_pred)
sns.displot(reg_pred-y_test, kind='kde')
#plt.show()
from sklearn.metrics import r2_score
score= r2_score(y_test, reg_pred)
print(score)
