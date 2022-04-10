import pandas as pd
import numpy as np
import math, re
import statistics

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
4. sibsp
5. parch 

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

df['Title'] = title
df['First & Middle Name'] = firstname
df['Surname'] = surname

#df = df.sort_values(by=['Name'])
#df.to_csv("clean_test.csv")

#Task 2
#if person < marriaged age, all sibsp is sib and all parch is parent, otherwise, it's children
#median age of marriage
MED_AGE_MARRY = 25

siblings = []
parents = []
children = []
spouses = []

for ind in df.index:
   if df['Age'][ind] < MED_AGE_MARRY or df['SibSp'][ind] == 0:
      siblings.append(df['SibSp'][ind])
   else:
      siblings.append('')

   if df['Age'][ind] < MED_AGE_MARRY or df['Parch'][ind] == 0:
      parents.append(df['Parch'][ind])
   else:
      parents.append(0)
   
df['Siblings'] = siblings
df['Parents'] = parents

for ind in df.index:
   children.append(df['Parch'][ind] - df['Parents'][ind])

df['Children'] = children

#if have children and sibsp >= 1, assume it is spouse
for ind in df.index:
   if df["Children"][ind] >= 1 and df['SibSp'][ind] >= 1:
      spouses.append(1)
      df['Siblings'] = df['SibSp'][ind] - 1
   else:
      spouses.append('')

df['Spouse'] = spouses

#if title is mrs. and have sibsp >=1, assume one of them is spouse
for ind in df.index:
   if 'Mrs' in df['Name'][ind] and df['SibSp'][ind] >= 1 and pd.isnull(df['Spouse'][ind]):
      if '(Hannah Wizosky)' in df['Name'][ind]:
         print('Mrs' in df['Name'][ind])
         print(df['SibSp'][ind] >= 1)
         print(pd.isnull(df['Spouse'][ind]))
      df['Spouse'][ind] = 1
      df['Siblings'][ind] = df['SibSp'][ind] - 1



#group people by last names, can tell mother and father by having parch and sibsp >= 1


#applies to males, if in same name group, there is a mrs. with your last name, then spouse +1, rest is sib



#task 3
#master is 0 - 13, halfway = 6
MED_MASTER_AGE = 6
for ind in df.index:
   if 'Master' in df['Name'][ind] and pd.isnull(df['Age'][ind]):
      df['Age'][ind] = MED_MASTER_AGE

#task 4
#apply median age to miss
miss_age = []
for ind in df.index:
   if 'Miss' in df['Name'][ind] and df['Age'][ind]:
      miss_age.append(df['Age'][ind])


for ind in df.index:
   if 'Miss' in df['Name'][ind] and pd.isnull(df['Age'][ind]):
      df['Age'] == statistics.median(miss_age)

#task 5
#if person is miss, sibsp must all be siblings, parch must be all parents
for ind in df.index:
   if 'Miss' in df['Name'][ind] and pd.isnull(df['Age'][ind]) and pd.isnull(df['Siblings'][ind]):
      df['Siblings'][ind] = df['SibSp'][ind]

#task 6
#can only have 1 spouse, if sibsp > 1, siblings will be sibsp-1
for ind in df.index:
   if df['SibSp'][ind] > 1 and pd.isnull(df['Siblings'][ind]):
      df['Siblings'] = df['SibSp'][ind] - 1



df = df.sort_values(by=['Name'])
df.to_csv("clean_test.csv")

print(set(title))


print(df.columns.tolist())

