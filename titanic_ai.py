import pandas as pd

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

    # Calculate Pr(survived) and Pr(dead)
    num_passengers = len(training_df.index)
    num_survivors  = training_df[["Survived"]].value_counts()[1]
    num_dead       = training_df[["Survived"]].value_counts()[0]

    pr_survived = num_survivors / num_passengers
    pr_dead     = num_dead / num_passengers


    # Compute CPTs for each passenger feature
    Pclass_df   = makeCPT("Pclass", list(training_df["Pclass"].unique()))
    Sex_df      = makeCPT("Sex", list(training_df["Sex"].unique()))
    SibSp_df    = makeCPT("SibSp", list(training_df["SibSp"].unique()))
    Parch_df    = makeCPT("Parch", list(training_df["Parch"].unique()))
    Embarked_df = makeCPT("Embarked", list(training_df["Embarked"].unique()))
    
    
    for index, row in training_df.iterrows(): 
        survived = row["Survived"] 

        Pclass_df[survived][row["Pclass"]] += 1
        Sex_df[survived][row["Sex"]] += 1 
        SibSp_df[survived][row["SibSp"]] += 1 
        Parch_df[survived][row["Parch"]] += 1 
        Embarked_df[survived][row["Embarked"]] += 1

    # Add Pr(Feature|Survived) and Pr(Feature|Died) to CPTs
    addConditionalProb(Pclass_df)
    addConditionalProb(Sex_df)
    addConditionalProb(SibSp_df)
    addConditionalProb(Parch_df)
    addConditionalProb(Embarked_df)

    print(Pclass_df)
    print(Sex_df)
    print(SibSp_df)
    print(Parch_df)
    print(Embarked_df)

def main(): 
    global training_df
    global test_df
    training_df = pd.read_csv("./datasets/train.csv")
    test_df     = pd.read_csv("./datasets/test.csv")
    
    naive_bayes_classifier()

    
if __name__ == "__main__": 
    main()