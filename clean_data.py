import pandas as pd
import numpy as np
import math, re

'''
headers:
[
   'PassengerId', 
   'Pclass', 
   'Name', 
   'Sex', 
   'Age', 
   'SibSp', 
   'Parch', 
   'Ticket', 
   'Fare', 
   'Cabin', 
   'Embarked'
]
'''

'''
Strategy:
1. Split name into Title col, First Name col, Middle name col, Last name col -- Done
2. Sort names by Last name
3. Clean up age data 
4. Clean up Cabin data
'''

'''
Declare a list that is to be converted into a column
address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']
 
 Using 'Address' as the column name
 and equating it to the list
df['Address'] = address

#df.to_csv("clean_test.csv", sep='\t')
'''


training_df = pd.read_csv("./datasets/train.csv")

test_df     = pd.read_csv("./datasets/test.csv")
df = pd.DataFrame(training_df)

#Task 1
names = [entry.replace('.', ',').split(", ") for entry in df["Name"].tolist()]
surname = [s[0] for s in names]
title = [t[1] for t in names]
firstname = [f[2] for f in names]

#Task 2

print(set(title))


print(df.columns.tolist())

