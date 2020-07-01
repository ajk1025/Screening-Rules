import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import coordinate_descent
import time
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


####Loading Dataset and Pre-processing
from sklearn.datasets import fetch_openml
print("Loading data...")
dataset = fetch_openml("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
Y = 2 * ((dataset.target == "AML") - 0.5)

n, p = X.shape

X = StandardScaler().fit_transform(X)
X = X / math.sqrt(n)

Y = Y.reshape((n,1))
Y = StandardScaler().fit_transform(Y)
Y = Y / math.sqrt(n)

B_0 = np.zeros((p, 1))

no_lambdas = 100
lambda_max = coordinate_descent.lambda_max(X, Y)[0]
grid_of_lambdas = np.linspace(lambda_max - 0.0001, 0, num = no_lambdas, endpoint=False)


####Measuring the number of features eliminated as a function of Lambda for each screening rule.

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




####Time Trials-> Clocking the total time of computation over a grid of Lambdas for each of the screening rules.  For greater accuracy, times are averaged over 4 trials.
#There are three different implementations of the Gap SAFE Test-> standard version, active set version, and working set version.

no_loops = 4
times_screened_coordinate_descent = [0]*no_lambdas
times_screened_sphere = [0]*no_lambdas
times_screened_dome = [0]*no_lambdas
dynamic_sphere_times = [0]*no_lambdas
dynamic_dome_times = [0]*no_lambdas
sequential_1_times = [0]*no_lambdas
sequential_2_times = [0]*no_lambdas
gap_safe_times = [0]*no_lambdas
gap_safe_active_set_times = [0]*no_lambdas
gap_safe_working_set_times = [0]*no_lambdas



time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	for j in range(no_lambdas):	
		time_start_within = time.perf_counter()
		B_0 = coordinate_descent.coordinate_descent(X, B_0, Y, grid_of_lambdas[j])
		time_elapsed_within = (time.perf_counter() - time_start_within)
		times_screened_coordinate_descent[j] += time_elapsed_within / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for coordinate descent is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	for j in range(no_lambdas):	
		time_start_within = time.perf_counter()
		B_0 = coordinate_descent.basic_sphere_test(X, B_0, Y, grid_of_lambdas[j])[1]
		time_elapsed_within = (time.perf_counter() - time_start_within)
		times_screened_sphere[j] += time_elapsed_within / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the basic sphere is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	for j in range(no_lambdas):
		time_start_within = time.perf_counter()
		B_0 = coordinate_descent.default_dome_test(X, B_0, Y, grid_of_lambdas[j])[1]	
		time_elapsed_within = (time.perf_counter() - time_start_within)
		times_screened_dome[j] += time_elapsed_within / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the default dome is:", time_elapsed)


time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.sequential_screening_1(X, B_0, Y, grid_of_lambdas)[2]
	for j in range(no_lambdas):
		sequential_1_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for screening test 1 is:", time_elapsed)


time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.sequential_screening_2(X, B_0, Y, grid_of_lambdas)[2]
	for j in range(no_lambdas):
		sequential_2_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for screening test 2 is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.dynamic_screening_sphere(X, B_0, Y, grid_of_lambdas)[1]
	for j in range(no_lambdas):
		dynamic_sphere_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the dynamic sphere is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.dynamic_screening_dome(X, B_0, Y, grid_of_lambdas)[1]
	for j in range(no_lambdas):
		dynamic_dome_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the dynamic dome is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.gap_safe_rule(X, B_0, Y, grid_of_lambdas)[1]
	for j in range(no_lambdas):
		gap_safe_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the gap safe rule is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.gap_safe_rule_active_set(X, B_0, Y, grid_of_lambdas)[1]
	for j in range(no_lambdas):
		gap_safe_active_set_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the active set gap safe rule is:", time_elapsed)

time_start = time.perf_counter()
for w in range(no_loops):
	B_0 = np.zeros((p, 1))
	solution = coordinate_descent.gap_safe_rule_working_set(X, B_0, Y, grid_of_lambdas)[1]
	for j in range(no_lambdas):
		gap_safe_working_set_times[j] += solution[j] / no_loops
time_elapsed = (time.perf_counter() - time_start)
print("The total time for the working set gap safe rule is:", time_elapsed)


plt.xlabel(r'$\lambda / \lambda_{max}$')
plt.ylabel("Time (seconds)")
plt.plot(grid_of_lambdas / lambda_max, times_screened_coordinate_descent, "k-", linewidth=1, label='Coordinate Descent')
plt.plot(grid_of_lambdas / lambda_max, times_screened_sphere, "b-", linewidth=1, label='Basic Sphere Screening Test')
plt.plot(grid_of_lambdas / lambda_max, times_screened_dome, "r-", linewidth=1, label='Default Dome Screening Test')
plt.plot(grid_of_lambdas / lambda_max, sequential_1_times, '-', color='orange', linewidth=1, label='Sequential Screening Test 1')
plt.plot(grid_of_lambdas / lambda_max, sequential_2_times, '-', color='yellow', linewidth=1, label='Sequential Screening Test 2')
plt.plot(grid_of_lambdas / lambda_max, dynamic_sphere_times, '-', color='rebeccapurple', linewidth=1, label='Dyanmic Sphere')
plt.plot(grid_of_lambdas / lambda_max, dynamic_dome_times, "c-", linewidth=1, label='Dynamic Dome')
plt.plot(grid_of_lambdas / lambda_max, gap_safe_times, '-', color='forestgreen', linewidth=1, label='Gap Safe')
plt.plot(grid_of_lambdas / lambda_max, gap_safe_active_set_times, '-', color='chartreuse', linewidth=1, label='Gap Safe (Active Set)')
plt.plot(grid_of_lambdas / lambda_max, gap_safe_working_set_times, '-', color='fuchsia', linewidth=1, label='Gap Safe (Working Set)')
plt.legend()
plt.yscale('log')
plt.show()
