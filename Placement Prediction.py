#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Load the dataset
data=pd.read_csv('collegePlace.csv')


# In[3]:


data.head()


# In[4]:


# checking the dataype of the parameters(Columns)
data.info()


# In[5]:


# To know columns
data.columns


# In[6]:


# Descriptive Analysis
data.describe()


# In[7]:


data.mean()


# In[8]:


#Checking null values
data.isnull().any()


# In[9]:


data.isnull().sum()


# In[10]:


#BarPlot
plt.bar(data['Age'],data['PlacedOrNot'])


# In[11]:


#BarPlot
plt.bar(data['Gender'],data['PlacedOrNot'])


# In[12]:


#BarPlot
plt.bar(data['Stream'],data['PlacedOrNot'])


# In[13]:


#BarPlot
plt.bar(data['Internships'],data['PlacedOrNot'])


# In[14]:


#BarPlot
plt.bar(data['CGPA'],data['PlacedOrNot'])


# In[15]:


#BarPlot
plt.bar(data['Hostel'],data['PlacedOrNot'])


# In[16]:


#BarPlot
plt.bar(data['HistoryOfBacklogs'],data['PlacedOrNot'])


# In[17]:


#Pie Chart
plt.pie(data['PlacedOrNot'],autopct='%.2f')


# In[18]:


plt.plot(data['PlacedOrNot'],data['Internships'],marker='o')
plt.plot(data['PlacedOrNot'],data['HistoryOfBacklogs'],marker='x')
plt.xlabel('Chance_of_Admit')
plt.ylabel('TOEFL and GRE Scores')


# In[19]:


plt.plot(data['PlacedOrNot'],data['Internships'],marker='o')
plt.plot(data['PlacedOrNot'],data['HistoryOfBacklogs'],marker='x')
plt.xlabel('Chance_of_Admit')
plt.ylabel('TOEFL and GRE Scores')


# In[20]:


plt.plot(data['PlacedOrNot'],data['CGPA'],marker='o')
plt.plot(data['PlacedOrNot'],data['HistoryOfBacklogs'],marker='x')
plt.xlabel('Chance_of_Admit')
plt.ylabel('TOEFL and GRE Scores')


# In[21]:


#Pair Plot
sns.pairplot(data)


# In[22]:


#Heat Map
hm=data.corr()
sns.heatmap(hm)


# In[23]:


#Label Encoding to convert categorical columns to numerical columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[24]:


data.Gender=le.fit_transform(data.Gender)


# In[25]:


data.Stream=le.fit_transform(data.Stream)


# In[26]:


#Splitting into dependent and independent variables
x=data.drop(columns=['PlacedOrNot'],axis=1)
y=data['PlacedOrNot']


# In[27]:


#Checking Outliers
sns.boxplot(x)


# In[28]:


#Calculating quartiles for x_independent
quantile = x.quantile(q=[0.25,0.75])
quantile


# In[29]:


#IQR
IQR = quantile.iloc[1] - quantile.iloc[0]
IQR


# In[30]:


#calculating upper extreme
upper_extreme = quantile.iloc[1] + (1.5*IQR)
upper_extreme


# In[31]:


#calculating lower extreme
lower_extreme = quantile.iloc[0] - (1.5*IQR)
lower_extreme


# In[32]:


#removing outliers from the extracted numeric columns

removed_outliers = x[(x >=lower_extreme)&(x <=upper_extreme)]
removed_outliers


# In[33]:


removed_outliers.to_csv('file1.csv')


# In[34]:


#Finding null values after removing outliers
removed_outliers.isnull().any()


# In[35]:


#Replacing null values
removed_outliers['Age'].fillna(removed_outliers['Age'].mean(),inplace=True)
removed_outliers['Gender'].fillna(removed_outliers['Gender'].mean(),inplace=True)
removed_outliers['Internships'].fillna(removed_outliers['Internships'].mean(),inplace=True)
removed_outliers['HistoryOfBacklogs'].fillna(removed_outliers['HistoryOfBacklogs'].mean(),inplace=True)


# In[36]:


#Checking whether null values are removed
removed_outliers.isnull().sum()


# In[37]:


removed_outliers


# In[38]:


#Removed outliers Boxplot
sns.boxplot(removed_outliers)


# # Scaling

# In[39]:


name=removed_outliers.columns
name


# In[40]:


#Normalisation
from sklearn.preprocessing import MinMaxScaler


# In[41]:


scale=MinMaxScaler()


# In[42]:


X_scaled=scale.fit_transform(removed_outliers)


# In[43]:


X=pd.DataFrame(X_scaled,columns=name)


# In[44]:


X


# In[45]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state = 9)


# In[46]:


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix
# models = [DecisionTreeClassifier(),RandomForestClassifier(),XGBClassifier(),KNeighborsClassifier()]
# model_name=["Decision Tree","Random Forest","XgBoost","K-Nearest Neighbors"]


# In[47]:


# model_scores=[]
# for model in models:
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test) 
#     model_scores.append(accuracy_score(y_test, y_pred))


# In[48]:


# for i in range(len(model_name)):
#     print(model_name[i],str(":"),model_scores[i])


# In[49]:


from xgboost import XGBClassifier
modelxgb=XGBClassifier()

modelxgb.fit(x_train,y_train)

acc_score8 = modelxgb.score(x_test, y_test)
print("model score: %.3f" % acc_score8)

y_pred8=modelxgb.predict(x_test)


# In[50]:


x_train


# In[51]:


# modelxgb.predict([[0.6667,0.0,0.6,0.5,0.75,1.0,0.0]])

new_data = pd.DataFrame({
    'Age': [20],
    'Gender': [0],
    'Stream': [1],
    'Internships': [1],
    'CGPA': [6],
    'Hostel': [1],
    'HistoryOfBacklogs': [1],
})
predicted_label = modelxgb.predict(new_data)
predicted_label[0]


# In[52]:


import pickle
pickle.dump(modelxgb,open("model.pkl","wb"))


# In[ ]:




