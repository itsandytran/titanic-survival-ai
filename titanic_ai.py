import pandas as pd

def linear_regression(): 
    return 

def naive_bayes_classifier(): 
    # Calculate Pr(survived) and Pr(dead)
    num_passengers = len(training_df.index)
    num_survivors  = training_df[["Survived"]].value_counts()[1]
    num_dead       = training_df[["Survived"]].value_counts()[0]

    pr_survived = num_survivors / num_passengers
    pr_dead     = num_dead / num_passengers


    # Calculate conditional probabilities for each passenger feature 
    pclass_dict = {
        'pclass': [1, 2, 3],
        'survived': [0, 0, 0],
        'dead': [0, 0, 0]
    }   
    
    for index, row in training_df.iterrows():
        survived = row["Survived"] 
        pclass   = row["Pclass"]
        
        if survived: 
            pclass_dict['survived'][pclass - 1] += 1
        else:
            pclass_dict['dead'][pclass - 1] += 1

    pclass = pd.DataFrame(data=pclass_dict)
    
    pclass_given_survived = []
    pclass_given_dead     = [] 
    for index, row in pclass.iterrows(): 
        pclass_and_survived = row["survived"]
        pclass_and_dead     = row["dead"]

        pclass_given_survived.append(pclass_and_survived / num_survivors)
        pclass_given_dead.append(pclass_and_dead / num_dead) 

    pclass["pclass_given_survived"] = pclass_given_survived
    pclass["pclass_given_dead"]     = pclass_given_dead

    print(pclass)


def main(): 
    global training_df
    global test_df
    training_df = pd.read_csv("./datasets/train.csv")
    test_df     = pd.read_csv("./datasets/test.csv")
    
    naive_bayes_classifier()

    
if __name__ == "__main__": 
    main()