import numpy as np
from matplotlib import pyplot as plt
from shapely import geometry, ops
import math
import random

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

""" Helpers """


def read_dataset(filename):
    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


def plot_grid(cell_percentages):
    max_lat = -8.58
    max_long = 41.18
    min_lat = -8.68
    min_long = 41.14

    background_image = plt.imread('porto.png')

    fig, ax = plt.subplots()
    ax.imshow(background_image, extent=[min_lat, max_lat, min_long, max_long], zorder=1)

    rec = [(min_lat, min_long), (min_lat, max_long), (max_lat, max_long), (max_lat, min_long)]
    nx, ny = 4, 5  # number of columns and rows  4,5

    polygon = geometry.Polygon(rec)
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [geometry.LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)]
    vertical_splitters = [geometry.LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters

    result = polygon
    for splitter in splitters:
        result = geometry.MultiPolygon(ops.split(result, splitter))

    grids = list(result.geoms)

    for grid_index, grid in enumerate(grids):
        x, y = grid.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        count = cell_percentages[grid_index]
        count = round(count, 2)

        centroid = grid.centroid
        ax.annotate(str(count) + '%', (centroid.x, centroid.y), color='black', fontsize=12,
                    ha='center', va='center', zorder=3)

    plt.show()


# You can define your own helper functions here. #

def sum_vectors(perturbed_values):
    sum_vector = [0] * 20
    for ii in range(20):
        for jj in range(len(perturbed_values)):
            sum_vector[ii] += perturbed_values[jj][ii]
    return sum_vector


def calculate_average_error(sum_vector, estimated_vector):
    agg_error = 0
    for ii in range(20):
        agg_error += abs(sum_vector[ii] - estimated_vector[ii])
    return agg_error / len(sum_vector)


def create_sum_vector(values):
    sum_vector = [0] * 20
    for ii in range(len(values)):
        sum_vector[values[ii] - 1] += 1
    return sum_vector


### HELPERS END ###

""" Functions to implement """

# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    probability_arr = [0] * 20
    values_arr = list(range(1, 21))
    values_arr.remove(val)
    values_arr.insert(0, val)
    d = 20
    p = math.pow(math.e, epsilon) / (math.pow(math.e, epsilon) + (d - 1))
    q = 1 / (math.pow(math.e, epsilon) + (d - 1))
    probability_arr[0] = p
    for ii in range(1, 20):
        probability_arr[ii] = probability_arr[ii - 1] + q 
    random_val = random.random()
    for ii in range(20):
        if random_val < probability_arr[ii]:
            return values_arr[ii]


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    d = 20
    p = math.pow(math.e, epsilon) / (math.pow(math.e, epsilon) + (d - 1))
    q = 1 / (math.pow(math.e, epsilon) + (d - 1))
    n = len(perturbed_values)
    estimated_vector = [0] * 20
    sum_vector = create_sum_vector(perturbed_values)
    for ii in range(20):
        estimated_vector[ii] = (sum_vector[ii] - n * q) / (p - q)
    return list(np.array(estimated_vector) * 100 / n)


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    n = len(dataset)
    sum_vector = list(np.array(create_sum_vector(dataset)) * 100 / n)
    perturbed_values = []
    for ii in range(len(dataset)):
        perturbed_val = perturb_grr(dataset[ii], epsilon)
        perturbed_values.append(perturbed_val)
    estimated_vector = estimate_grr(perturbed_values, epsilon)
    return calculate_average_error(sum_vector, estimated_vector)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    bit_vector = [0] * 20
    bit_vector[val - 1] = 1
    return bit_vector


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    p = math.pow(math.e, (epsilon / 2)) / (math.pow(math.e, (epsilon / 2)) + 1)
    for ii in range(20):
        if not (random.random() < p):
            encoded_val[ii] = int(not encoded_val[ii])
    return encoded_val


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    p = math.pow(math.e, (epsilon / 2)) / (math.pow(math.e, (epsilon / 2)) + 1)
    q = 1 - p
    n = len(perturbed_values)
    sum_vector = sum_vectors(perturbed_values)
    estimated_vector = [0] * 20
    for ii in range(20):
        estimated_vector[ii] = (sum_vector[ii] - n * q) / (p - q)
    return list((np.array(estimated_vector) * 100) / n)


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    actual_values = []
    perturbed_values = []
    for ii in range(len(dataset)):
        encoded_val = encode_rappor(dataset[ii])
        actual_values.append(encoded_val)
        perturbed_val = perturb_rappor(encoded_val, epsilon)
        perturbed_values.append(perturbed_val)
    n = len(actual_values)
    estimated_vector = estimate_rappor(perturbed_values, epsilon)
    sum_vector = list(np.array(sum_vectors(actual_values)) * 100 / n)
    return calculate_average_error(sum_vector, estimated_vector)


# OUE

# TODO: Implement this function!
def encode_oue(val):
    bit_vector = [0] * 20
    bit_vector[val - 1] = 1
    return bit_vector


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    p0 = math.pow(math.e, epsilon) / (math.pow(math.e, epsilon) + 1)
    p1 = 1/2
    for ii in range(20):
        if encoded_val[ii] == 0:
            p = p0
        else:
            p = p1
        if not (random.random() < p):
            encoded_val[ii] = int(not encoded_val[ii])
    return encoded_val


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    n = len(perturbed_values)
    sum_vector = sum_vectors(perturbed_values)
    estimated_vector = [0] * 20
    for ii in range(20):
        estimated_vector[ii] = (2 * ((math.pow(math.e, epsilon) + 1) * sum_vector[ii] - n)) / (math.pow(math.e, epsilon) - 1)
    return list((np.array(estimated_vector) * 100) / n)


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    actual_values = []
    perturbed_values = []
    for ii in range(len(dataset)):
        encoded_val = encode_oue(dataset[ii])
        actual_values.append(encoded_val)
        perturbed_val = perturb_oue(encoded_val, epsilon)
        perturbed_values.append(perturbed_val)
    n = len(actual_values)
    estimated_vector = estimate_oue(perturbed_values, epsilon)
    plot_grid(estimated_vector)
    sum_vector = list(np.array(sum_vectors(actual_values)) * 100 / n)
    return calculate_average_error(sum_vector, estimated_vector)


def main():
    dataset = read_dataset("taxi-locations.dat")

    print("GRR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))
            
    

if __name__ == "__main__":
    main()
