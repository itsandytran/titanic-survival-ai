import pandas as pd

def linear_regression(): 
    return 

def naive_bayes_classifier(): 
    num_passengers = len(training_df.index)
    num_survivors  = training_df[["Survived"]].value_counts()[1]
    num_dead       = training_df[["Survived"]].value_counts()[0]

    pr_survived = num_survivors / num_passengers
    pr_dead     = num_dead / num_passengers

    print(pr_survived)
    print(pr_dead)

def main(): 
    global training_df
    global test_df
    training_df = pd.read_csv("./datasets/train.csv")
    test_df     = pd.read_csv("./datasets/test.csv")
    
    naive_bayes_classifier()

    
if __name__ == "__main__": 
    main()