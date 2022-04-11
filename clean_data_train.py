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


#task 3
#master is 0 - 13, halfway = 6
MED_MASTER_AGE = 6
for ind in df.index:
   if 'Master' in df['Name'][ind] and pd.isnull(df['Age'][ind]):
      df.at[ind, 'Age'] = MED_MASTER_AGE

# #task 4
# #apply median age to miss
miss_age = []
for ind in df.index:
   if 'Miss' in df['Name'][ind] and not pd.isna(df['Age'][ind]):
      miss_age.append(df['Age'][ind])

for ind in df.index:
   if 'Miss' in df['Name'][ind] and pd.isna(df['Age'][ind]):
      df.at[ind, 'Age'] = statistics.median(miss_age)

#rest of the mr ages
mr_age = []
for ind in df.index:
   if 'Mr' in df['Name'][ind] and not pd.isna(df['Age'][ind]):
      mr_age.append(df['Age'][ind])

for ind in df.index:
   if 'Mr' in df['Name'][ind] and pd.isna(df['Age'][ind]):
      df.at[ind, 'Age'] = statistics.median(mr_age)


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
      df.at[ind, 'Siblings'] = df['SibSp'][ind] - 1
   elif df['SibSp'][ind] == 0:
      spouses.append(0)
   else:
      spouses.append('')

df['Spouse'] = spouses

#if person is <25, assume no spouse
for ind in df.index:
   if df['Age'][ind] < MED_AGE_MARRY and df['Spouse'][ind] == '':
      df.at[ind, 'Spouse'] = 0


#if title is mrs. and have sibsp >=1, assume one of them is spouse
for ind in df.index:
   if 'Mrs' in df['Name'][ind] and df['SibSp'][ind] >= 1 and df['Spouse'][ind]=='':
      df.at[ind, 'Spouse'] = 1
      df.at[ind, 'Siblings'] = df['SibSp'][ind] - 1

for ind in df.index:
   if 'Dr' in df['Name'][ind] and pd.isna(df['Age'][ind]):
      df.at[ind, 'Age'] = 42

#group people by last names, can tell mother and father by having parch and sibsp >= 1
#make names dictionary, match 
# families = {}
# for ind in df.index:
#    if df['Surname'][ind] not in families:
#       families[df['Surname'][ind]] = [(df.iloc[ind], ind)]
#    else:
#       families[df['Surname'][ind]].append((df.iloc[ind], ind))


# for key in families.keys():
#    #groups of two
#    f = families[key]
#    if len(f) == 2:
#       for person in f:
#          if df['Spouse'][f[0][1]] == 1 or df['Spouse'][f[1][1]] == 1:
#             print("hello")
#             df.at[f[0][1], 'Spouse'] == 1
#             df.at[f[1][1], 'Spouse'] == 1
#             df.at[f[0][1], 'Siblings'] == df['SibSp'][f[0][1]] - 1
#             df.at[f[1][1], 'Siblings'] == df['SibSp'][f[1][1]] - 1
      

for ind in df.index:
   if 'Mr' in df['Name'][ind] and df['Spouse'][ind] == '' and df['SibSp'][ind] >= 1:
      df.at[ind, 'Spouse'] = 1
      df.at[ind, 'Siblings'] = df['SibSp'][ind] - 1


# #applies to males, if in same name group, there is a mrs. with your last name, then spouse +1, rest is sib


#task 5
#if person is miss, sibsp must all be siblings, parch must be all parents
for ind in df.index:
   if 'Miss' in df['Name'][ind] and pd.isnull(df['Age'][ind]) and pd.isnull(df['Siblings'][ind]):
      df.at[ind, 'Siblings'] = df['SibSp'][ind]

# #task 6
#can only have 1 spouse, if sibsp > 1, siblings will be sibsp-1
for ind in df.index:
   if df['SibSp'][ind] > 1 and pd.isnull(df['Siblings'][ind]):
      df.at[ind, 'Siblings'] = df['SibSp'][ind] - 1

#final clean
for ind in df.index:
   if df['Siblings'][ind] == '' or df['Spouse'][ind] == '':
      df.at[ind, 'Siblings'] = df['SibSp'][ind]
      df.at[ind, 'Spouse'] = 0

#edit

df = df.sort_values(by=['Name'])
df.to_csv("clean_training.csv")

print(set(title))


print(df.columns.tolist())

