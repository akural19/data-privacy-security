import sys
import random

import numpy as np
import pandas as pd
import copy
import math

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings 

warnings.filterwarnings("ignore")

############################# Label Flipping ##################################

def label_flipping_helper(X_train, y_train, p):
    columns = ["variance", "skewness", "curtosis", "entropy", "class", "index"]
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    df = pd.concat([X_train, y_train], axis = 1)
    df["index"] = df.index
    arr = np.array(df)
    np.random.shuffle(arr)
    temp_df = pd.DataFrame(arr, columns = columns)
    num_rows = math.ceil(len(df) * p)
    temp_df["original_class"] = temp_df["class"]
    for ii in range(num_rows):
        temp_df.iloc[ii, 4] = int(not temp_df.iloc[ii, 4])
    temp_df = temp_df.sort_values(by = "index", ascending = True)
    temp_df.drop("index", axis = 1, inplace = True)
    return temp_df.reset_index(drop = True)

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth = 5, random_state = 0)
    elif model_type == "SVC":
        model = SVC(C = 0.5, kernel = 'poly', random_state = 0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    else:
        print("Invalid ML Model!")
        return -1
    
    acc_scores = []
    for ii in range(100):
        temp_df = label_flipping_helper(X_train, y_train, p)
        temp_X_train = temp_df.iloc[:, 0:4]
        temp_y_train = temp_df.iloc[:, 4]
        model.fit(temp_X_train, temp_y_train)
        prediction = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, prediction))
    return sum(acc_scores) / len(acc_scores)

########################### Label Flipping Defense ############################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    temp_df = label_flipping_helper(X_train, y_train, p)
    scaler = StandardScaler()
    new_df = temp_df.copy()
    new_df.iloc[:, 0:5] = scaler.fit_transform(new_df.iloc[:, 0:5])
    
    model = LocalOutlierFactor(contamination = p)
    outlier_indices = list(np.where(model.fit_predict(new_df.iloc[:, 0:5]) == -1)[0])
    attacked_indices = list(temp_df[temp_df["class"] != temp_df["original_class"]].index)
    
    correct_guess_num = len(list(set(outlier_indices).intersection(set(attacked_indices))))
    num_flipped_label = len(attacked_indices)
    
    print("Out of %d flipped data points, %d were correctly identified." %(num_flipped_label, correct_guess_num))


############################# Evasion Attack ##################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    
    step = 0.15
    while True:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    for zz in range(2):
                        if ii == 0:
                            modified_example[0] -= step
                        elif ii == 1:
                            modified_example[0] += step
                        if jj == 0:
                            modified_example[1] -= step
                        elif jj == 1:
                            modified_example[1] += step
                        if kk == 0:
                            modified_example[2] -= step
                        elif kk == 1:
                            modified_example[2] += step
                        if zz == 0:
                            modified_example[3] -= step
                        elif zz == 1:
                            modified_example[3] += step
        pred_class = trained_model.predict([modified_example])[0]
        if pred_class != actual_class:
            return modified_example
        modified_example = actual_example.copy()
        step *= 1.05


def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


########################## Transferability ####################################

def transferability_helper(trained_model, transfer_models, actual_examples):
    for actual_example in actual_examples:
        modified_examples = []
        modified_examples.append(evade_model(trained_model, actual_example))
    model1_score = 0
    model2_score = 0
    for ii in range(len(modified_examples)):
        actual_example = actual_examples[ii]
        modified_example = modified_examples[ii]
        actual_label = transfer_models[0].predict([actual_example])[0]
        modified_label = transfer_models[0].predict([modified_example])[0]
        if modified_label != actual_label:
            model1_score += 1   
        actual_label = transfer_models[1].predict([actual_example])[0]
        modified_label = transfer_models[1].predict([modified_example])[0]
        if modified_label != actual_label:
            model2_score += 1
    return model1_score, model2_score
    

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    x11, x12 = transferability_helper(DTmodel, [LRmodel, SVCmodel], actual_examples)
    x21, x22 = transferability_helper(SVCmodel, [LRmodel, DTmodel], actual_examples)
    x31, x32 = transferability_helper(LRmodel, [DTmodel, SVCmodel], actual_examples)

    print("Out of 40 adversarial examples crafted to evade DT :")
    print("-> %d of them transfer to LR." %x11)
    print("-> %d of them transfer to SVC." %x12)

    print("Out of 40 adversarial examples crafted to evade LR :")
    print("-> %d of them transfer to DT." %x21)
    print("-> %d of them transfer to SVC." %x22)

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print("-> %d of them transfer to DT." %x31)
    print("-> %d of them transfer to LR." %x32)

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: %.3f' %accuracy_score(y_test, DEC_predict))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: %.3f' %accuracy_score(y_test, LR_predict))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: %.3f' %accuracy_score(y_test, SVC_predict))

    # # Label flipping attack executions:
    # print("#"*50)
    # print("Label flipping attack executions:")
    # model_types = ["DT", "LR", "SVC"]
    # p_vals = [0.05, 0.10, 0.20, 0.40]
    # for model_type in model_types:
    #     for p in p_vals:
    #         acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
    #         print("Accuracy of poisoned %s %.2f : %.3f" %(model_type, p, acc))

    # # Label flipping defense executions:
    # print("#" * 50)
    # print("Label flipping defense executions:")
    # p_vals = [0.05, 0.10, 0.20, 0.40]
    # for p in p_vals:
    #     print("Results with p=", str(p), ":")
    #     label_flipping_defense(X_train, y_train, p)

    # Evasion attack executions:
    # print("#"*50)
    # print("Evasion attack executions:")
    # trained_models = [myDEC, myLR, mySVC]
    # model_types = ["DT", "LR", "SVC"]
    # num_examples = 40
    # for a, trained_model in enumerate(trained_models):
    #     total_perturb = 0.0
    #     for i in range(num_examples):
    #         actual_example = X_test[i]
    #         adversarial_example = evade_model(trained_model, actual_example)
    #         if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
    #             print("Evasion attack not successful! Check function: evade_model.")
    #         perturbation_amount = calc_perturbation(actual_example, adversarial_example)
    #         total_perturb = total_perturb + perturbation_amount
    #     print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])



if __name__ == "__main__":
    main()


