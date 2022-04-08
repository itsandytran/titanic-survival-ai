import pandas as pd
import numpy as np
import math, re

MAX_LOOP_COUNT = 5000

def clean(): 
    return 

def logistic_regression(training_df, test_df): 
    # Normalizes the parametes in a dataframe containing only numerical series
    def normalize_parameters(df):
        sex = {"male" : 0,
               "female" : 1}
        df["Sex"] = df["Sex"].map(sex)
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df = (df - df.min()) / (df.max()-df.min())

        return df

    # Applies the logistic regression equation hw(x) to all passengers given their params
    def generate_predictions(params_list, weights, bias):
        predictions = []
        for params in params_list:
            predictions.append(predict_survival(params, weights, bias))
        
        return predictions

    # Implementation of logistic regression equation hw(x) for a single passenger
    def predict_survival(params, weights, bias):
        expn = math.exp((np.sum(params * weights) + bias) * -1)
        decision_value = 1 / (1 + expn)
        return decision_value

    # Returns total loss given a set of predictions range[0,1] and outcomes range{0,1}
    def calculate_total_loss(predictions, outcomes):
        total_loss = 0
        for i in range(len(predictions)):
            total_loss += calculate_loss(predictions[i], outcomes[i])

        return total_loss * -1

    # Implementation of the loss function for a single prediction-outcome pair
    def calculate_loss(prediction, outcome):
        return math.log(prediction) if outcome else math.log(1 - prediction)

    # Performs gradient descent algorithm on a set of weight parameters
    def gradient_descent(weights, bias, params_list, outcomes, alpha):
        new_weights = weights.copy()
        iteration = 0
        predictions = []
        min = False
        prev_t_loss = None

        while not(min) and iteration <= MAX_LOOP_COUNT:
            for i in range(len(new_weights)):
                summation = 0
                for j in range(len(params_list)):
                    surv = predict_survival(params_list[j], weights, bias)
                    summation += (surv - outcomes[j]) * params_list[j][i]

                new_weights[i] -= alpha * summation
            
            summation = 0
            for i in range(len(params_list)):
                surv = predict_survival(params_list[i], weights, bias)
                summation += (surv - outcomes[i])

            bias = alpha * summation
            weights = new_weights
            predictions = generate_predictions(params_list, weights, bias)
            total_loss = calculate_total_loss(predictions, outcomes)

            #if(iteration % 100 == 0):
                #print(f'After iteration {iteration}: \nweights = {weights}\nbias = {bias}\ntotal_loss={total_loss}')
            
            if (total_loss == prev_t_loss):
                min = True
            else:
                prev_t_loss = total_loss
            
            iteration += 1
        
        return (weights, bias, predictions)

    """ ==================================== PARAMETERS ====================================
    Index           |         0          1          2            3            4          5
    Attribute       |    Pclass        Sex        Age        SibSp        Parch       Fare
    Value Type      |       int        int      float          int          int      float
    Value Range     |     
    (non-normalized)|     [1, 3]      [0-1]     [0-80]        [0-8]        [0-9]    [0-512]

    Description:
    Pclass - Ticket class. 1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class
    Sex - 0 = male, 1 = female
    SibSp - Number of siblings and/or spouses
    ParCh - Number of parents and/or children
    Fare - Cost of passenger fare (not adjusted for inflation)
    """

    submission_df = test_df["PassengerId"].to_frame()
    # Setting up the datasets

    outcomes = training_df["Survived"].to_numpy()
    training_df = training_df.drop(["PassengerId", "Survived", "Name", "Ticket" , "Cabin", "Embarked"], axis=1)
    tr_params_list = normalize_parameters(training_df).to_numpy()

    test_df = test_df.drop(["PassengerId", "Name", "Ticket" , "Cabin", "Embarked"], axis=1)
    test_params_list = normalize_parameters(test_df).to_numpy()

    # Inital weight and bias
    init_weights = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    init_bias = -2.0

    # Perform gradient_descent until convergence
    result = gradient_descent(init_weights, init_bias, tr_params_list, outcomes, 0.02)
    weights = np.array(result[0])
    bias = result[1]

    predictions = np.array(generate_predictions(test_params_list, weights, bias))
    survival = np.around(predictions).astype('i')
    submission_df["Survived"] = survival

    submission_df.to_csv("submission_logreg.csv", index=False)

def naive_bayes_classifier(training_df, test_df): 
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
    pr_died     = num_dead / num_passengers

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
    
    # Count number of survivors/dead per feature
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

    # Test
    PassengerIds = [] 
    Survived = [] 
    for index, row in test_df.iterrows(): 

        pr_Pclass_given_survived = Pclass_df["feature_given_survived"][row["Pclass"]]
        pr_Sex_given_survived = Sex_df["feature_given_survived"][row["Sex"]]
        pr_SibSp_given_survived = SibSp_df["feature_given_survived"][row["SibSp"]]
        pr_Embarked_given_survived = Embarked_df["feature_given_survived"][row["Embarked"]]
        pr_Cabin_given_survived = Cabin_df["feature_given_survived"][getCabinLetter(row["Cabin"])]
        pr_Age_given_survived = Age_df["feature_given_survived"][getAgeGroup(row["Age"])]
        #pr_Title_given_survived = Title_df["feature_given_survived"][getTitle(row["Name"])]
        
        pr_Pclass_given_died = Pclass_df["feature_given_died"][row["Pclass"]]
        pr_Sex_given_died = Sex_df["feature_given_died"][row["Sex"]]
        pr_SibSp_given_died = SibSp_df["feature_given_died"][row["SibSp"]]
        #pr_Parch_given_died = Parch_df["feature_given_died"][row["Parch"]]
        pr_Embarked_given_died = Embarked_df["feature_given_died"][row["Embarked"]]
        pr_Cabin_given_died = Cabin_df["feature_given_died"][getCabinLetter(row["Cabin"])]
        pr_Age_given_died = Age_df["feature_given_died"][getAgeGroup(row["Age"])]
        #pr_Title_given_died = Title_df["feature_given_died"][getTitle(row["Name"])]
        
        pr_survived_given_features = pr_Pclass_given_survived * \
                                     pr_Sex_given_survived * \
                                     pr_SibSp_given_survived * \
                                     pr_Embarked_given_survived * \
                                     pr_Cabin_given_survived * \
                                     pr_Age_given_survived
        
        pr_died_given_features = pr_Pclass_given_died * \
                                 pr_Sex_given_died * \
                                 pr_SibSp_given_died * \
                                 pr_Embarked_given_died * \
                                 pr_Cabin_given_died * \
                                 pr_Age_given_died
        

        if pr_survived_given_features > pr_died_given_features: 
            Survived.append(1)
        elif pr_survived_given_features < pr_died_given_features:
            Survived.append(0)
        else: 
            print("MAYDAYYYY") 
        
    submission_df = pd.DataFrame({"PassengerId": test_df["PassengerId"].values,
                                  "Survived"   : Survived
                                 })

    submission_df.to_csv("submission_naivebayes.csv", index=False)

def main(): 
    training_df = pd.read_csv("./datasets/train.csv")
    test_df     = pd.read_csv("./datasets/test.csv")
    
    naive_bayes_classifier(training_df, test_df)
    logistic_regression(training_df, test_df)

    
if __name__ == "__main__": 
    main()