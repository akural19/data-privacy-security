import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv
import re

from numpy import sqrt, exp

''' Functions to implement '''

# TODO: Implement this function!
def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    pattern = r'\b(\d+)-(\d+)\b'
    dataset["year"] = None
    dataset["month"] = None
    for ii in range(len(dataset)):
        match = re.search(pattern, dataset.loc[ii, "date"])
        dataset.loc[ii, "year"] = match.group(1)
        dataset.loc[ii, "month"] = match.group(2)
    return dataset.drop("date", axis = 1)

def filter_dataset(dataset, year, state):
    dataset = dataset[(dataset["year"] == year) & (dataset["state"] == state)]
    return dataset.reset_index(drop = True)

# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):
    dataset = filter_dataset(dataset, year, state)
    plt.figure()
    plt.bar(dataset["month"], dataset["positive"])
    title_string = "[NO DP] Positive Test Case for State {0} in year {1}".format(state, year)
    plt.title(title_string)
    plt.xticks(rotation = 90)
    plt.show()
    return list(dataset["positive"])

def add_laplace_noise(dataset, epsilon, sensitivity):
    b = sensitivity / epsilon
    mean = 0
    for ii in range(len(dataset)):
        noise = np.random.laplace(mean, b)
        dataset.loc[ii, "positive"] += noise
    return dataset

# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):
    dataset = filter_dataset(dataset, year, state)
    sensitivity = N
    dataset = add_laplace_noise(dataset, epsilon, sensitivity)
    plt.figure()
    plt.bar(dataset["month"], dataset["positive"])
    title_string = "[Epsilon ({0})-DP] Positive Test Case for State {1} in year {2}".format(epsilon, state, year)
    plt.title(title_string)
    plt.xticks(rotation = 90)
    plt.show()
    return list(dataset["positive"])

# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    agg_error = 0
    for ii in range(len(actual_hist)):
        agg_error = abs(actual_hist[ii] - noisy_hist[ii])
    return agg_error / len(actual_hist)

# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):
    avg_errors = []
    sensitivity = N
    actual_hist = get_histogram(dataset, state, year)
    for epsilon in eps_values:
        error_sum = 0
        for ii in range(10):
            noisy_hist = get_dp_histogram(dataset, state, year, epsilon, sensitivity)
            AvgErr = calculate_average_error(actual_hist, noisy_hist)
            error_sum += AvgErr
        avg_errors.append((error_sum / 10))
    return avg_errors

# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):
    avg_errors = []
    actual_hist = get_histogram(dataset, state, year)
    for sensitivity in N_values:
        error_sum = 0
        for ii in range(10):
            noisy_hist = get_dp_histogram(dataset, state, year, epsilon, sensitivity)
            AvgErr = calculate_average_error(actual_hist, noisy_hist)
            error_sum += AvgErr
        avg_errors.append((error_sum / 10))
    return avg_errors


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #

def calculate_probability(dataset, sensitivity, epsilon):
    dataset["probability"] = None
    denominator = 0
    for ii in range(len(dataset)):
        power = (epsilon * dataset.loc[ii, "death"]) / (2 * sensitivity)
        val = math.pow(math.e, power)
        dataset.loc[ii, "probability"] = val
        denominator += val
    dataset["probability"] /= denominator
    return dataset

def define_ranges(dataset):
    dataset["upper_range"] = None
    for ii in range(len(dataset)):
        if ii == 0 :
            dataset.loc[ii, "upper_range"] = dataset.loc[ii, "probability"]
        else:
            dataset.loc[ii, "upper_range"] = dataset.loc[(ii - 1), "upper_range"] + dataset.loc[ii, "probability"]
    return dataset   

# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    sensitivity = 1
    dataset = filter_dataset(dataset, year, state)
    dataset = calculate_probability(dataset, sensitivity, epsilon)
    dataset = define_ranges(dataset)
    random_val = random.random()
    for ii in range(len(dataset)):
        if random_val < dataset.loc[ii, "upper_range"]:
            return dataset.loc[ii, "month"]


# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):
    accuracy_arr = []
    dataset = filter_dataset(dataset, year, state)
    correct_result = dataset.loc[list(dataset["death"]).index(max(dataset["death"])), "month"]
    for epsilon in epsilon_list:
        counter = 0
        for ii in range(1000):
            month = max_deaths_exponential(dataset, state, year, epsilon)
            if month == correct_result:
                counter += 1
        accuracy_arr.append(counter / 10)
    return accuracy_arr
    
    
# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)
    
    state = "TX"
    year = "2020"

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
          print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy (%) = ", exponential_experiment_result[i])



if __name__ == "__main__":
    main()
