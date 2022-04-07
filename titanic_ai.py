import pandas as pd
import numpy as np
import re

def linear_regression(): 
    return 

def naive_bayes_classifier(): 
    def makeCPT(feature, feature_types): 
        dictionary = {feature: feature_types} 
        dictionary[1] = [0] * len(feature_types) # Initialize number of survivors to 0
        dictionary[0] = [0] * len(feature_types) # Initialize number of dead to 0

        dataframe = pd.DataFrame(data=dictionary)   # Convert dictionary to dataframe
        dataframe.set_index(feature, inplace=True)  # Let dataframe's index be the first column
        return dataframe

    def addConditionalProb(CPT): 
        feature_given_survived = []
        feature_given_died     = [] 

        for index, row in CPT.iterrows(): 
            feature_and_survived = row[1]
            feature_and_died     = row[0]

            feature_given_survived.append(feature_and_survived / num_survivors)
            feature_given_died.append(feature_and_died / num_dead)
        
        CPT["feature_given_survived"] = feature_given_survived
        CPT["feature_given_died"]     = feature_given_died

    def getCabinLetter(cabin): 
        # If cabin is a string, return the first letter
        if isinstance(cabin, str): 
            return cabin[0]
        
        # If cabin is an array, return a list of unique cabin first letters
        elif isinstance(cabin, np.ndarray):
            cabin_list = list(cabin)
            for i in range(len(cabin_list)):        
                if isinstance(cabin_list[i], str):
                    cabin_list[i] = cabin_list[i][0]
            return list(set(cabin_list))
    
        return cabin
    
    def getAgeGroup(age): 
        # If age is a float, return age group
        if isinstance(age, float): 
            if age < 15: 
                return "Children"
            elif age < 25: 
                return "Youth"
            elif age < 65: 
                return "Adults"
            else: 
                return "Seniors"
        return age 

    def getTitle(name): 
        # If name is a string, return their title 
        if isinstance(name, str):
            split = name.split(",")[1].split()
            for s in split: 
                if "." in s:
                    return s

        # If name is an array, return a list of unique name titles (Mr. Mrs. Dr. etc)
        if isinstance(name, np.ndarray):
            titles = set()
            for n in name: 
                split = n.split(",")[1].split()
                for s in split: 
                    if "." in s:
                        titles.add(s)

            return list(titles)
        
        return name

    # Calculate Pr(survived) and Pr(dead)
    num_passengers = len(training_df.index)
    num_survivors  = training_df[["Survived"]].value_counts()[1]
    num_dead       = training_df[["Survived"]].value_counts()[0]

    pr_survived = num_survivors / num_passengers
    pr_dead     = num_dead / num_passengers

    # Initialize CPTs for each passenger's (categorical/discrete) features
    Pclass_df   = makeCPT("Pclass", list(training_df["Pclass"].unique()))
    Sex_df      = makeCPT("Sex", list(training_df["Sex"].unique()))
    SibSp_df    = makeCPT("SibSp", list(training_df["SibSp"].unique()))
    Parch_df    = makeCPT("Parch", list(training_df["Parch"].unique()))
    Embarked_df = makeCPT("Embarked", list(training_df["Embarked"].unique()))
    
    # Initialize CPTs for each passenger's continuous features
    Cabin_df    = makeCPT("Cabin", getCabinLetter(training_df["Cabin"].unique()))
    Age_df      = makeCPT("Age", ["Children", "Youth", "Adults", "Seniors"])
    
    Title_df = makeCPT("Title", getTitle(training_df["Name"].unique()))
    
    """ ========= FARES ==========
    fares = list(training_df["Fare"].unique())
    fares.sort()
    print(fares)
    """
    

    for index, row in training_df.iterrows(): 
        survived = row["Survived"] 
        
        # Discrete features 
        Pclass_df[survived][row["Pclass"]] += 1
        Sex_df[survived][row["Sex"]] += 1 
        SibSp_df[survived][row["SibSp"]] += 1 
        Parch_df[survived][row["Parch"]] += 1 
        Embarked_df[survived][row["Embarked"]] += 1

        # Continuous features
        Cabin_df[survived][getCabinLetter(row["Cabin"])] += 1
        Age_df[survived][getAgeGroup(row["Age"])] += 1
        Title_df[survived][getTitle(row["Name"])] += 1

    # Add Pr(Feature|Survived) and Pr(Feature|Died) to CPTs
    addConditionalProb(Pclass_df)
    addConditionalProb(Sex_df)
    addConditionalProb(SibSp_df)
    addConditionalProb(Parch_df)
    addConditionalProb(Embarked_df)
    addConditionalProb(Cabin_df)
    addConditionalProb(Age_df)
    addConditionalProb(Title_df)
    
    """
    print(Pclass_df)
    print(Sex_df)
    print(SibSp_df)
    print(Parch_df)
    print(Embarked_df)
    print(Cabin_df)
    print(Age_df)
    print(Title_df)
    """

def main(): 
    global training_df
    global test_df
    training_df = pd.read_csv("./datasets/train.csv")
    test_df     = pd.read_csv("./datasets/test.csv")
    
    naive_bayes_classifier()

    
if __name__ == "__main__": 
    main()