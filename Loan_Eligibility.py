#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd

train = pd.read_csv(r'C:\Users\HP\Downloads\Loan_predict\train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv(r'C:\Users\HP\Downloads\Loan_predict\test_Y3wMUE5_7gLdaTN.csv')
train.info()

train_original=train.copy()
test_original=test.copy()


# In[66]:


import matplotlib.pyplot as plt
train['Dependents'].value_counts(normalize=True).plot.bar(title='Dependents')
plt.show()
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.show()
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()

plt.subplots_adjust(hspace=1)
"""Most of the applicants don't have any dependents.
Around 80% of the applicants are Graduate."""


# In[67]:


train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
plt.show()
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.show()
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.show()
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.show()


"""80% of applicants in the dataset are male.
Around 65% of the applicants in the dataset are married.
Around 15% of applicants in the dataset are self-employed"""


# In[68]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[69]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[70]:


import numpy as np
train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log']=np.log(test['LoanAmount'])

train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1)
y = train.Loan_Status


X = pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.2,stratify =y,random_state =42)


# In[73]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression()

pred_cv = model.predict(x_cv)
print(pred_cv)

print("Accuracy: ", metrics.accuracy_score(y_cv,pred_cv))
print("Confusion Matrix on Test Data")
pd.crosstab(y_cv, pred_cv, rownames=['True'], colnames=['Predicted'], margins=True)


# In[74]:


pred_test = model.predict(test)
print(pred_test)


submission = pd.read_csv(r'C:\Users\HP\Downloads\Loan_predict\submission.csv')
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv(r'C:\Users\HP\Downloads\Loan_predict\submission.csv')


# In[ ]:




