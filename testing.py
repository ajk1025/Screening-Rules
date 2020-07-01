#This file contains applies the screening rules defined in the coordinate_descent.py file to a synthetic data set containing 20 data points and 1000 parameters.
#Features were drawn from independent Gaussian distributions, and Y values wee computed as Y = X^T Beta + Gaussian error.  The true number of relevant features is 10. 
#The feature matrix is normalied so that columns have mean zero and L2 norm of 1.  The Y vector is also centered and normalized in the same way.  
#For each screening rule, we measure the number of features it eliminates as a function of Lambda. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import coordinate_descent

####Generating Synthetic Data
n, p = 20, 1000
no_relevant_features = int(min(n, p/2))
true_beta = np.zeros((p, 1))
for j in range(no_relevant_features):
	true_beta[j] = 0.1 + (1 / (j+1))
X = np.random.normal(0, 1, (n,p))
X = StandardScaler().fit_transform(X)
X = X / math.sqrt(n)
error = np.random.normal(0, 0.2, (n,1))
error = StandardScaler(with_std=False).fit_transform(error)
Y = np.matmul(X, true_beta) + error
scale_factor = coordinate_descent.L2(Y)
Y = Y / coordinate_descent.L2(Y)
B = np.random.rand(p,1)
B_0 = np.zeros((p, 1))
no_lambdas = 100


lambda_max = coordinate_descent.lambda_max(X, Y)[0]
print("lambda_max is ", lambda_max)
grid_of_lambdas = np.linspace(lambda_max - .0001, 0, num = no_lambdas, endpoint=False)

####Actual testing

true_zeros = []
B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
for j in grid_of_lambdas:	
	true_zeros.append(p - np.count_nonzero(coordinate_descent.coordinate_descent(X, B_0, Y, j)))
time_elapsed = (time.perf_counter() - time_start)
print("The total time for coordinate descent is:", time_elapsed)

no_features_screened_basic_sphere = []
B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
for j in grid_of_lambdas:	
	no_features_screened_basic_sphere.append(coordinate_descent.basic_sphere_test(X, B_0, Y, j)[0])
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the basic sphere test is:", time_elapsed)

no_features_screened_default_dome = []
B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
for j in grid_of_lambdas:
	no_features_screened_default_dome.append(coordinate_descent.default_dome_test(X, B_0, Y, j)[0])	
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the default dome test is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
no_features_screened_sequential_1 = coordinate_descent.sequential_screening_1(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for screening test 1 is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
no_features_screened_sequential_2 = coordinate_descent.sequential_screening_2(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for screening test 2 is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
dynamic_sphere = coordinate_descent.dynamic_screening_sphere(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the dynamic sphere test is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
dynamic_dome = coordinate_descent.dynamic_screening_dome(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the dynamic dome test is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
gap_safe = coordinate_descent.gap_safe_rule(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the gap safe rule is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
gap_safe_active_set = coordinate_descent.gap_safe_rule_active_set(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the gap safe rule active set is:", time_elapsed)

B_0 = np.zeros((p, 1))
time_start = time.perf_counter()
gap_safe_working_set = coordinate_descent.gap_safe_rule_working_set(X, B_0, Y, grid_of_lambdas)[0]
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the gap safe rule working set is:", time_elapsed)

plt.xlabel(r'$\lambda / \lambda_{max}$')
plt.ylabel("Number of Features Screened")
plt.plot(grid_of_lambdas / lambda_max, true_zeros, "k:", linewidth=1, label='True Number of Inactive Features')
plt.plot(grid_of_lambdas / lambda_max, no_features_screened_basic_sphere, "b-", linewidth=1, label='Basic Sphere Screening Test')
plt.plot(grid_of_lambdas / lambda_max, no_features_screened_default_dome, "r-", linewidth=1, label='Default Dome Screening Test')
plt.plot(grid_of_lambdas / lambda_max, no_features_screened_sequential_1, '-', color='orange', linewidth=1, label='Sequential Screening Test 1')
plt.plot(grid_of_lambdas / lambda_max, no_features_screened_sequential_2, '-', color='yellow', linewidth=1, label='Sequential Screening Test 2')
plt.plot(grid_of_lambdas / lambda_max, dynamic_sphere, "m-", linewidth=1, label='Dynamic Sphere Test')
plt.plot(grid_of_lambdas / lambda_max, dynamic_dome, "c-", linewidth=1, label='Dynamic Dome Test')
plt.plot(grid_of_lambdas / lambda_max, gap_safe, '-', color='forestgreen', linewidth=1, label='Gap Safe Test')
#plt.plot(grid_of_lambdas / lambda_max, gap_safe_active_set, '-', color='chartreuse', linewidth=1, label='Gap Safe Test Active Set')
#plt.plot(grid_of_lambdas / lambda_max, gap_safe_working_set, '-', color='fuchsia', linewidth=1, label='Gap Safe Test Working Set')
plt.legend()
plt.show()

