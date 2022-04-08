import pandas as pd
import numpy as np
import re

def clean(): 
    return 

def linear_regression(): 
    return 

def naive_bayes_classifier(training_df, test_df): 
    """ Employs the Naive Bayes Theorem to make predictions on the survival of passengers 
        of the shipwrecked Titanic based on their passenger profile (e.g. name, sex, class, etc).
        Outputs a CSV containing these predictions.

    Parameters: 
    training_df (np.ndarray) : dataframe containing training data
    test_df     (np.ndarray) : dataframe on which this algorithm will predict passenger survival
	"""
    def makeCPT(feature, feature_options): 
        """ Returns a dataframe representing the Conditional Probability Table (CPT)
            from the given feature (e.g. sex) and options within that feature (e.g. male, female)
            against 2 dependent variables: 1 (passenger survived) and 0 (passenger died)

            e.g.    Sex      1    0      
                    ---      -    -
                    Male     0    0  
                    Female   0    0  <--- # passengers who are female and 0 (died) = 0

            We initialize the counts for each conjunction to 0.  
        """
        dictionary = {feature: feature_options} 
        dictionary[1] = [0] * len(feature_options)  # Initialize number of survivors with the feature to 0
        dictionary[0] = [0] * len(feature_options)  # Initialize number of dead with the feature to 0

        dataframe = pd.DataFrame(data=dictionary)   # Convert dictionary to dataframe
        dataframe.set_index(feature, inplace=True)  # Let dataframe's index be the first column
        return dataframe

    def addConditionalProb(CPT): 
        """ Adds two columns to the provided Conditional Probability Table (CPT)
            representing the conditional probabilities Pr(feature | survived) and Pr(feature | died)
        """
        feature_given_survived = []
        feature_given_died     = [] 

        # Loop through rows of the CPT 
        for index, row in CPT.iterrows(): 
            num_survivors_with_feature = row[1] 
            num_dead_with_feature      = row[0] 

            feature_given_survived.append(laPlaceSmoothing(num_survivors_with_feature, num_survivors))
            feature_given_died.append(laPlaceSmoothing(num_dead_with_feature, num_dead))

        
        CPT["feature_given_survived"] = feature_given_survived
        CPT["feature_given_died"]     = feature_given_died

    def getCabinLetter(cabin): 
        """ Returns the first letter of each passenger's cabin. """

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
        """ Returns the age group of the provided age. 
            The age groups are taken from the statscan website.
        """
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
        """ ** EDIT TO BE REMOVED AFTER HAROLD'S TITLE IMPLEMENTATION
            Returns the title from a person's name (e.g. Dr., Mrs., Mme. etc)
        """
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
    
    def laPlaceSmoothing(num_deadOrAlive_w_feature, num_deadOrAlive, num_features=8, alpha=1): 
        """ This is a smoothing technique to circumvent the zero frequency problem.
            It's possible to encounter feature options in the test set that aren't present 
            in the CPTs because they aren't available in the training set.
            Rather than returning a Pr(feature | survived) = 1 or 0, we can use this smoothing technique

            LaPlace Smoothing equation: 
                                        num_survivors_with_feature + alpha
            Pr(feature | survived) =  ------------------------------------------
                                    num_survivors + alpha * number_of_features
        """
        return (num_deadOrAlive_w_feature + alpha) / (num_deadOrAlive + alpha * num_features)
    
    def getPrFeatureGivenDeadOrAlive(CDT, feature_option, feature_given_deadOrAlive):
        # If this feature option (e.g. 9 kids) is not in the CDT generated by training data
        # Let num_deadOrAlive_w_feature = 0 and apply LaPlace Smoothing
        if feature_option not in CDT.index: 

            # Determine if we need number of survivors or dead 
            num_deadOrAlive = 0
            if feature_given_deadOrAlive == "feature_given_survived": 
                num_deadOrAlive = num_survivors
            else: # "feature_given_died"
                num_deadOrAlive = num_dead

            return laPlaceSmoothing(0, num_deadOrAlive)
        
        return CDT[feature_given_deadOrAlive][feature_option]

    # Calculate Pr(survived) and Pr(dead)
    num_passengers = len(training_df.index)
    num_survivors  = training_df[["Survived"]].value_counts()[1]
    num_dead       = training_df[["Survived"]].value_counts()[0]
    pr_survived    = num_survivors / num_passengers
    pr_died        = num_dead / num_passengers

    # Initialize CPTs for passenger's features
    Pclass_df   = makeCPT("Pclass", list(training_df["Pclass"].unique()))
    Sex_df      = makeCPT("Sex", list(training_df["Sex"].unique()))
    SibSp_df    = makeCPT("SibSp", list(training_df["SibSp"].unique()))
    Parch_df    = makeCPT("Parch", list(training_df["Parch"].unique()))
    Embarked_df = makeCPT("Embarked", list(training_df["Embarked"].unique()))
    Cabin_df    = makeCPT("Cabin", getCabinLetter(training_df["Cabin"].unique()))
    Age_df      = makeCPT("Age", ["Children", "Youth", "Adults", "Seniors"])
    Title_df    = makeCPT("Title", getTitle(training_df["Name"].unique()))

    # Count the number of survivors or dead per feature 
    for index, row in training_df.iterrows(): 
        survived = row["Survived"] 
        
        Pclass_df[survived][row["Pclass"]] += 1                 # Pclass
        Sex_df[survived][row["Sex"]] += 1                       # Sex
        SibSp_df[survived][row["SibSp"]] += 1                   # Siblings/Spouses ** EDIT
        Parch_df[survived][row["Parch"]] += 1                   # Parents/Children ** EDIT
        Embarked_df[survived][row["Embarked"]] += 1             # Embarked
        Cabin_df[survived][getCabinLetter(row["Cabin"])] += 1   # Cabin letter
        Age_df[survived][getAgeGroup(row["Age"])] += 1          # Age group
        Title_df[survived][getTitle(row["Name"])] += 1          # Title

    # Add Pr(Feature|Survived) and Pr(Feature|Died) to CPTs
    addConditionalProb(Pclass_df)
    addConditionalProb(Sex_df)
    addConditionalProb(SibSp_df)
    addConditionalProb(Parch_df)
    addConditionalProb(Embarked_df)
    addConditionalProb(Cabin_df)
    addConditionalProb(Age_df)
    addConditionalProb(Title_df)        

    # Make survival predictions on passengers in test data (test_df)
    Survival_predictions = [] 
    for index, row in test_df.iterrows(): 

        pr_Pclass_given_survived   = getPrFeatureGivenDeadOrAlive(Pclass_df  , row["Pclass"]                , "feature_given_survived")
        pr_Sex_given_survived      = getPrFeatureGivenDeadOrAlive(Sex_df     , row["Sex"]                   , "feature_given_survived")
        pr_SibSp_given_survived    = getPrFeatureGivenDeadOrAlive(SibSp_df   , row["SibSp"]                 , "feature_given_survived")
        pr_Parch_given_survived    = getPrFeatureGivenDeadOrAlive(Parch_df   , row["Parch"]                 , "feature_given_survived")
        pr_Embarked_given_survived = getPrFeatureGivenDeadOrAlive(Embarked_df, row["Embarked"]              , "feature_given_survived")
        pr_Cabin_given_survived    = getPrFeatureGivenDeadOrAlive(Cabin_df   , getCabinLetter(row["Cabin"]) , "feature_given_survived")
        pr_Age_given_survived      = getPrFeatureGivenDeadOrAlive(Age_df     , getAgeGroup(row["Age"])      , "feature_given_survived")
        pr_Title_given_survived    = getPrFeatureGivenDeadOrAlive(Title_df   , getTitle(row["Name"])        , "feature_given_survived")
        
        pr_Pclass_given_died   = getPrFeatureGivenDeadOrAlive(Pclass_df  , row["Pclass"]                , "feature_given_died")
        pr_Sex_given_died      = getPrFeatureGivenDeadOrAlive(Sex_df     , row["Sex"]                   , "feature_given_died")
        pr_SibSp_given_died    = getPrFeatureGivenDeadOrAlive(SibSp_df   , row["SibSp"]                 , "feature_given_died")
        pr_Parch_given_died    = getPrFeatureGivenDeadOrAlive(Parch_df   , row["Parch"]                 , "feature_given_died")
        pr_Embarked_given_died = getPrFeatureGivenDeadOrAlive(Embarked_df, row["Embarked"]              , "feature_given_died")
        pr_Cabin_given_died    = getPrFeatureGivenDeadOrAlive(Cabin_df   , getCabinLetter(row["Cabin"]) , "feature_given_died")
        pr_Age_given_died      = getPrFeatureGivenDeadOrAlive(Age_df     , getAgeGroup(row["Age"])      , "feature_given_died")
        pr_Title_given_died    = getPrFeatureGivenDeadOrAlive(Title_df   , getTitle(row["Name"])        , "feature_given_died")
        
        # Compute the posterior probabilities 
        # Pr(survived | features) and Pr(died | features)
        pr_survived_given_features = pr_Pclass_given_survived * \
                                     pr_Sex_given_survived * \
                                     pr_SibSp_given_survived * \
                                     pr_Parch_given_survived * \
                                     pr_Embarked_given_survived * \
                                     pr_Cabin_given_survived * \
                                     pr_Age_given_survived * \
                                     pr_Title_given_survived * \
                                     pr_survived
        
        pr_died_given_features = pr_Pclass_given_died * \
                                 pr_Sex_given_died * \
                                 pr_SibSp_given_died * \
                                 pr_Parch_given_died * \
                                 pr_Embarked_given_died * \
                                 pr_Cabin_given_died * \
                                 pr_Age_given_died * \
                                 pr_Title_given_died * \
                                 pr_died
        
        # Make prediction by choosing the larger of the two posterior probabilities
        if pr_survived_given_features > pr_died_given_features: 
            Survival_predictions.append(1)
        else:
            Survival_predictions.append(0)
    
    # Create submission CSV (columns are PassengerId and Survived prediction)
    submission_df = pd.DataFrame({"PassengerId": test_df["PassengerId"].values,
                                  "Survived"   : Survival_predictions })
    submission_df.to_csv("submission_naivebayes.csv", index=False)

def main(): 
    training_df = pd.read_csv("./datasets/train.csv")
    test_df     = pd.read_csv("./datasets/test.csv")
    
    naive_bayes_classifier(training_df, test_df)

if __name__ == "__main__": 
    main()