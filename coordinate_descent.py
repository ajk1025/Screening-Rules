import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time
import copy
import operator
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


####Global Variables
X_Y_Dot_Product = []
X_Y_Dot_Product_2 = []
X_n_Dot_Product = []
universal_duality_gap = 1E-6
no_iterations = 10000

####Helper Functions

def L1(x):
	sum = 0
	for i in range(len(x)):
		sum += abs(x[i])
	return(sum)

def L2(x):
	sum = 0
	for i in range(len(x)):
		sum += (x[i])**2
	return(math.sqrt(sum))

def L_Infinity(x):
	maximum = 0
	for el in x:
		if abs(el) > maximum:
			maximum = abs(el)
	return(maximum)

def sign(x):
	if x > 0:
		return(1)
	if x == 0:
		return(0)
	if x < 0:
		return(-1)

def soft_thresholding(x, t):
	if abs(x) < t:
		return(0)
	else:
		sol = sign(x)*(abs(x) - t)
		return(sol)

#Computes the minimum value of lambda which forces the solution vector to be identically zero
def lambda_max(X, Y):
	global X_Y_Dot_Product 
	temp_max = 0
	for j in range(len(X[0])):
		current = float(abs(np.dot(X[:,j], Y)))
		X_Y_Dot_Product.append(current)
		if current > temp_max:
			temp_max = current
			if np.dot(X[:,j], Y) > 0:
				most_correlated = X[:,j]
			else:
				most_correlated = -X[:,j]
	return(temp_max, most_correlated)

#Produces a dual feasible point by scaling the residual 
def dual_feasible_scaler(X, B, Y, Lambda):
	residual = Y - np.matmul(X, B)
	option1 = max(np.matmul(np.transpose(Y), residual) / (Lambda * (L2(residual)**2)), -1 / (L_Infinity(np.matmul(np.transpose(X), residual))))
	option2 = 1 / (L_Infinity(np.matmul(np.transpose(X), residual)))
	alpha = min(option1, option2)
	dual_feasible = alpha*residual
	return(dual_feasible)

#A faster version for when the residual has already been computed
def dual_feasible_scaler_modified(X, B, Y, Lambda, Residual, reduced_X):
	Residual = Residual.reshape(len(Y), 1)
	q = L_Infinity(np.matmul(np.transpose(reduced_X), Residual))
	option1 = max(np.matmul(np.transpose(Y), Residual) / (Lambda * (L2(Residual)**2)), -1 / q)
	option2 = 1 / q
	alpha = min(option1, option2)
	dual_feasible = alpha*Residual
	return(dual_feasible)

#Computes value of the Lasso objective
def compute_lasso_error(X, B, Y, Lambda):
	error = 0.5*((L2(Y - np.matmul(X, B)))**2) + Lambda*L1(B)
	return(error)

#A faster version for when the residual has already been computed
def compute_lasso_error_modified(X, B, Y, Lambda, Residual):
	Residual = Residual.reshape(len(Y), 1)
	error = 0.5*((L2(Residual))**2) + Lambda*L1(B)
	return(error)

#Computes the Duality Gap for the Lasso Problem
def duality_gap(X, B, Y, Lambda):
	dual_point = dual_feasible_scaler(X, B, Y, Lambda)
	dual_objective = 0.5*(L2(Y)**2) - (Lambda**2 / 2)* (L2(dual_point - (1 / Lambda)*Y)**2)
	gap = compute_lasso_error(X, B, Y, Lambda) - dual_objective
	return(gap)

#A faster version for when residual has already been computed
def duality_gap_modified(X, B, Y, Lambda, Residual, reduced_X):
	dual_point = dual_feasible_scaler_modified(X, B, Y, Lambda, Residual, reduced_X)
	dual_objective = 0.5*(1) - (Lambda**2 / 2)* (L2(dual_point - (1 / Lambda)*Y)**2)
	gap = compute_lasso_error(X, B, Y, Lambda) - dual_objective
	return(gap)

#Performs cross-validation to determine the optimal degree of regularization
def Optimal_Lambda(X, Y):
	Lambda_max = lambda_max(X, Y)[0]
	grid_of_lambdas = np.logspace(-6, 0, num=1000) / len(X)
	model = LassoCV(cv=10, alphas=grid_of_lambdas, max_iter=100000, tol=0.0001).fit(X, Y)
	best_alpha = model.alpha_ *len(X) 
	return(best_alpha)

#Solve the Lasso Problem using the sklearn implementation -> used in testing process to verify that my code was returning the correct answers
def accuracy_check(X, Y, Lambda):
	clf = linear_model.Lasso(alpha=(Lambda / len(X)))
	clf.fit(X, Y)
	sol = clf.coef_
	sol = sol.reshape((len(X[0]),1)) 
	return(sol)

####Coordinate Descent Implementation

def coordinate_update(X, B, Y, Lambda, j, Residual):
	arg1 = B[j] + np.dot(X[:,j],Residual)
	arg2 = Lambda
	B[j] = soft_thresholding(arg1, arg2)
	return(B)

def coordinate_descent(X, B, Y, Lambda): 
	Residual = Y - np.matmul(X, B)
	Residual = Residual.reshape(-1)
	for i in range(no_iterations):
		for k in range(len(B)):
			prev = copy.copy(B[k])
			B = coordinate_update(X, B, Y, Lambda, k, Residual)
			Residual += X[:,k]*(prev - B[k])
		if (i % 10) == 0:
			if len(X[0]) == 0:
				break
			if duality_gap(X, B, Y, Lambda) < universal_duality_gap:
				break
	return(B)

####Static Safe Rules

#General form of a sphere test
def sphere_test(X, B, Y, Lambda, c, r, *args):
	global X_Y_Dot_Product
	p = len(B)
	irrelevant_variables = []
	relevant_variables = []
	for j in range(p):
		if args:
			if X_Y_Dot_Product[j] / Lambda + r < 1:
				irrelevant_variables.append(j)
			else:
				relevant_variables.append(j)
		else:	
			if abs(np.dot(X[:,j], c)) + r < 1:
				irrelevant_variables.append(j)
			else:
				relevant_variables.append(j)
	reduced_X = np.delete(X, irrelevant_variables, 1)
	reduced_B = np.delete(B, irrelevant_variables, 0)
	reduced_solution = coordinate_descent(reduced_X, reduced_B, Y, Lambda)
	final_solution = np.zeros((p, 1))
	final_solution[relevant_variables] = reduced_solution
	no_irrelevant_variables = len(irrelevant_variables)
	return(no_irrelevant_variables, final_solution)

#"Basic sphere test" proposed by El Ghaoui et al. -> sphere center is Y / Lambda and radius is (1 / Lambda) - (1 / Lambda_max)
def basic_sphere_test(X, B, Y, Lambda):
	global X_Y_Dot_Product
	c = Y / Lambda
	Lambda_max = lambda_max(X, Y)[0]
	r = (1 / Lambda) - (1 / Lambda_max)
	return(sphere_test(X, B, Y, Lambda, c, r, 1))

#General form of a dome test
def dome_test(X, B, Y, Lambda, c, r, n, p, *args):
	global dome_time
	global X_n_Dot_Product
	global X_Y_Dot_Product_2
	psi = (np.matmul(np.transpose(n), c) - p) / r
	p = len(B)
	irrelevant_variables = []
	relevant_variables = []
	sqrt_exp = math.sqrt(abs(1 - psi**2))
	for j in range(p):
		if args:
			dot = X_n_Dot_Product[j]
		else:	
			dot = np.matmul(X[:,j], n)
		if dot < -psi:
			M1 = r
		else:
			M1 = -psi*r*dot + r*math.sqrt(abs(1 - dot**2))*sqrt_exp
		dot2 = -dot
		if dot2 < -psi:
			M2 = r
		else:
			M2 = -psi*r*dot2 + r*math.sqrt(abs(1 - (dot2)**2))*sqrt_exp
		if args:
			corr = X_Y_Dot_Product_2[j] / Lambda
		else:
			corr = np.matmul(X[:,j], c)
		if (M2 - 1 < corr) and (corr < 1 - M1):
			irrelevant_variables.append(j)
		else:
			relevant_variables.append(j)
	reduced_X = np.delete(X, irrelevant_variables, 1)
	reduced_B = np.delete(B, irrelevant_variables, 0)
	reduced_solution = coordinate_descent(reduced_X, reduced_B, Y, Lambda)
	final_solution = np.zeros((p, 1))
	final_solution[relevant_variables] = reduced_solution
	no_irrelevant_variables = len(irrelevant_variables)
	return(no_irrelevant_variables, final_solution)

#"Default dome test" proposed by Xiang et al. -> c and r borrowed from Basic Sphere Test, n = Feature vector most correlated with Y, p =1
def default_dome_test(X, B, Y, Lambda):
	global X_n_Dot_Product
	global X_Y_Dot_Product_2
	a, b = lambda_max(X,Y)
	c = Y / Lambda
	r = 1/Lambda - (1 / a)
	n = b
	p = 1
	if not len(X_n_Dot_Product):
		X_n_Dot_Product = np.matmul(np.transpose(X), n)
	if not len(X_Y_Dot_Product_2):
		X_Y_Dot_Product_2 = np.matmul(np.transpose(X), Y)
	return(dome_test(X, B, Y, Lambda, c, r, n, p, 1))

#### Sequential Screening Rules

#Sequential Screening test proposed by Xiang et al. -> dome test with c and r borrowed from Basic Sphere Test, n and p depend on solution of previous Lambda value
def sequential_screening_1(X, B, Y, Lambdas):
	total_times = []
	p = len(B)
	B_grid = []
	Theta_grid = []
	no_irrelevant_variables_grid = []
	time_start = time.perf_counter()
	B_0 = coordinate_descent(X, B, Y, Lambdas[0])
	B_grid.append(B_0)
	Theta_0 = (1 / Lambdas[0])*(Y - np.matmul(X, B))
	Theta_grid.append(Theta_0)
	no_irrelevant_variables_grid.append(p - 1)
	time_elapsed = (time.perf_counter() - time_start)
	total_times.append(time_elapsed)
	no_lambdas = len(Lambdas)
	for j in range(1, no_lambdas):
		time_start = time.perf_counter()
		c = Y / Lambdas[j]
		r = L2(c - Theta_grid[-1])
		if L2((Y / Lambdas[j-1]) - Theta_grid[-1]) != 0:
			n = (Y / Lambdas[j-1] - Theta_grid[-1]) / (L2((Y / Lambdas[j-1]) - Theta_grid[-1]))
			p = np.dot(np.transpose(n), Theta_grid[-1])
		A, B = dome_test(X, B, Y, Lambdas[j], c, r, n, p)
		no_irrelevant_variables_grid.append(A)
		B_grid.append(B)
		Theta_grid.append((1 / Lambdas[j])*(Y - np.matmul(X, B)))
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(no_irrelevant_variables_grid, B_grid, total_times)

#Sequential Screening 2 is the test proposed in Wang et al. -> Sphere Test with c = Dual solution for previous Lambda, r = |1 / Lambda_prev - 1 / Lambda|
def sequential_screening_2(X, B, Y, Lambdas):
	total_times = []
	time_start = time.perf_counter()
	p = len(B)
	B_grid = []
	Theta_grid = []
	no_irrelevant_variables_grid = []
	B_0 = np.zeros((p, 1))
	B_grid.append(B_0)
	Theta_0 = Y / Lambdas[0]
	Theta_grid.append(Theta_0)
	no_irrelevant_variables_grid.append(basic_sphere_test(X, B, Y, Lambdas[0])[0])
	time_elapsed = (time.perf_counter() - time_start)
	total_times.append(time_elapsed)
	no_lambdas = len(Lambdas)
	for j in range(1, no_lambdas):
		time_start = time.perf_counter()
		A, B = sphere_test(X, B, Y, Lambdas[j], Theta_grid[-1], abs((1/ Lambdas[j-1]) - (1 / Lambdas[j])))
		no_irrelevant_variables_grid.append(A)
		B_grid.append(B)
		Theta_grid.append((1 / Lambdas[j])*(Y - np.matmul(X, B)))
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(no_irrelevant_variables_grid, B_grid, total_times)

#### Dynamic Screening Rules

#Dynamic Screening Sphere Test proposed by Bonefoy et al.
def dynamic_screening_sphere(X, B, Y, Lambdas):
	total_times = []
	p = len(B)
	prod = np.matmul(np.transpose(X), Y)
	prod = np.absolute(prod)
	no_irrelevant_variables = []
	for L in Lambdas:
		time_start = time.perf_counter()
		reduced_X = X
		reduced_B = B
		irrelevant_variables = []
		relevant_variables = [i for i in range(p)]
		Residual = Y - np.matmul(X, B)
		Residual = Residual.reshape(-1)
		for j in range(no_iterations):
			if (j % 10) == 0:
				delete_set = []
				full_B = np.zeros((p, 1))
				full_B[relevant_variables] = reduced_B
				theta = dual_feasible_scaler(X, full_B, Y, L)
				r = L2(theta - Y / L)
				for k in relevant_variables:
					if (prod[k] / L) + r < 1:	
						irrelevant_variables.append(k)
						delete_set.append(k)
				for x in delete_set:
					relevant_variables.remove(x)
				reduced_X = np.delete(X, irrelevant_variables, 1)
				reduced_B = np.delete(full_B, irrelevant_variables, 0)
				if len(reduced_X[0]) == 0:
					break
				if duality_gap(reduced_X, reduced_B, Y, L) < universal_duality_gap:
					break 
			for k in range(len(reduced_B)):
				prev = copy.copy(reduced_B[k])
				reduced_B = coordinate_update(reduced_X, reduced_B, Y, L, k, Residual)
				Residual += reduced_X[:,k]*(prev - reduced_B[k])
		no_irrelevant_variables.append(len(irrelevant_variables))
		B = np.zeros((p, 1))
		B[relevant_variables] = reduced_B
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(no_irrelevant_variables, total_times)

#Dynamic Screening Dome Test proposed by Bonnefoy et al.
def dynamic_screening_dome(X, B, Y, Lambdas):
	total_times = []
	p = len(B)
	a, b = lambda_max(X,Y)
	total_irrelevant_variables = []
	for L in Lambdas:
		time_start = time.perf_counter()
		reduced_X = X
		reduced_B = B
		full_B = B
		c = Y / L
		irrelevant_variables = []
		relevant_variables = [i for i in range(len(B))]
		Residual = Y - np.matmul(X, B)
		Residual = Residual.reshape(-1)
		for j in range(no_iterations):
			if (j%10) == 0:
				delete_set = []
				full_B = np.zeros((len(B), 1))
				full_B[relevant_variables] = reduced_B
				theta = dual_feasible_scaler(X, full_B, Y, L)
				r = L2(theta - c)
				psi = (np.matmul(np.transpose(b), c) - 1) / r
				for k in relevant_variables:
					dot = np.matmul(X[:,k], b)
					if dot < -psi:
						M1 = r
					else:
						M1 = -psi*r*dot + r*math.sqrt(abs(1 - dot**2))*math.sqrt(abs(1 - psi**2))
					dot2 = -dot
					if dot2 < -psi:
						M2 = r
					else:
						M2 = -psi*r*dot2 + r*math.sqrt(abs(1 - (dot2)**2))*math.sqrt(abs(1 - psi**2))
					corr = np.matmul(X[:,k], c)
					if (M2 - 1 < corr) and (corr < 1 - M1) and psi < 1:
						irrelevant_variables.append(k)
						delete_set.append(k)
				for x in delete_set:
					relevant_variables.remove(x)
				reduced_X = np.delete(X, irrelevant_variables, 1)
				reduced_B = np.delete(full_B, irrelevant_variables, 0)
				if len(reduced_X[0]) == 0:
					break
				if duality_gap(reduced_X, reduced_B, Y, L) < universal_duality_gap:
					break 
			for k in range(len(reduced_B)):
				prev = copy.copy(reduced_B[k])
				reduced_B = coordinate_update(reduced_X, reduced_B, Y, L, k, Residual)
				Residual += reduced_X[:,k]*(prev - reduced_B[k])
		total_irrelevant_variables.append(len(irrelevant_variables))
		B = np.zeros((p, 1))
		B[relevant_variables] = reduced_B
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(total_irrelevant_variables, total_times)


#### Gap SAFE rule

#Basic Gap SAFE Test proposed by Fercoq et al.
def gap_safe_rule(X, B, Y, Lambdas):
	total_times = []
	p = len(B)
	total_irrelevant_variables = []
	for L in Lambdas:
		time_start = time.perf_counter()
		reduced_X = X
		reduced_B = B
		full_B = B
		irrelevant_variables = []
		relevant_variables = [x for x in range(len(B))]
		Residual = Y - np.matmul(X, B)
		Residual = Residual.reshape(-1)
		for j in range(no_iterations):
			if (j%10) == 0:
				delete_set = []
				full_B = np.zeros((len(B), 1))
				full_B[relevant_variables] = reduced_B
				theta = dual_feasible_scaler_modified(X, full_B, Y, L, Residual, reduced_X)
				c = theta
				r_big = L2(theta - Y / L)
				r_small = (1 / L) * math.sqrt(max((1 - L2(Residual)**2 - 2*L*L1(reduced_B)), 0))
				r = math.sqrt(abs(r_big**2 - r_small**2))
				for k in relevant_variables:
					if abs(np.dot(X[:,k], c)) + r < 1:
						irrelevant_variables.append(k)
						delete_set.append(k)
				for x in delete_set:
					relevant_variables.remove(x)
				reduced_X = np.delete(X, irrelevant_variables, 1)
				reduced_B = np.delete(full_B, irrelevant_variables, 0)
				if len(reduced_X[0]) == 0:
					break
				if duality_gap_modified(reduced_X, reduced_B, Y, L, Residual, reduced_X) < universal_duality_gap:
					break 
			for k in range(len(reduced_B)):
				prev = copy.copy(reduced_B[k])
				reduced_B = coordinate_update(reduced_X, reduced_B, Y, L, k, Residual)
				Residual += reduced_X[:,k]*(prev - reduced_B[k])
		total_irrelevant_variables.append(len(irrelevant_variables))
		B = np.zeros((p, 1))
		B[relevant_variables] = reduced_B
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(total_irrelevant_variables, total_times)

#Gap SAFE Test with active warm start; initialize by first solving Lasso problem restricted to support of previous Lambda value
def gap_safe_rule_active_set(X, B, Y, Lambdas):
	total_times = []
	p = len(B)
	total_irrelevant_variables = []
	for L in Lambdas:
		time_start = time.perf_counter()
		if L != Lambdas[0]:
			B[relevant_variables] = coordinate_descent(reduced_X, reduced_B, Y, L)
		reduced_X = X
		reduced_B = B
		full_B = B
		irrelevant_variables = []
		relevant_variables = [x for x in range(len(B))]
		Residual = Y - np.matmul(X, B)
		Residual = Residual.reshape(-1)
		for j in range(no_iterations):
			if (j%10) == 0:
				delete_set = []
				full_B = np.zeros((len(B), 1))
				full_B[relevant_variables] = reduced_B
				theta = dual_feasible_scaler_modified(X, full_B, Y, L, Residual, reduced_X)
				c = theta
				r_big = L2(theta - Y / L)
				r_small = (1 / L) * math.sqrt(max((1 - L2(Residual)**2 - 2*L*L1(reduced_B)), 0))
				r = math.sqrt(abs(r_big**2 - r_small**2))
				for k in relevant_variables:
					if abs(np.dot(X[:,k], c)) + r < 1:
						irrelevant_variables.append(k)
						delete_set.append(k)
				for x in delete_set:
					relevant_variables.remove(x)
				reduced_X = np.delete(X, irrelevant_variables, 1)
				reduced_B = np.delete(full_B, irrelevant_variables, 0)
				if len(reduced_X[0]) == 0:
					break
				if duality_gap_modified(reduced_X, reduced_B, Y, L, Residual, reduced_X) < universal_duality_gap:
					break 
			for k in range(len(reduced_B)):
				prev = copy.copy(reduced_B[k])
				reduced_B = coordinate_update(reduced_X, reduced_B, Y, L, k, Residual)
				Residual += reduced_X[:,k]*(prev - reduced_B[k])
		total_irrelevant_variables.append(len(irrelevant_variables))
		B = np.zeros((p, 1))
		B[relevant_variables] = reduced_B
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(total_irrelevant_variables, total_times)

def gap_safe_rule_active_warm_start(X, B, Y, Target_Lambda):
	total_times = []
	p = len(B)
	total_irrelevant_variables = []
	l_max = lambda_max(X, Y)[0]
	Lambdas = np.linspace(l_max, Target_Lambda, num = 5, endpoint=True)
	for L in Lambdas:
		if L != l_max:
			B[relevant_variables] = coordinate_descent(reduced_X, reduced_B, Y, L)
		reduced_X = X
		reduced_B = B
		irrelevant_variables = []
		relevant_variables = [x for x in range(len(B))]
		Residual = Y - np.matmul(X, B)
		Residual = Residual.reshape(-1)
		for j in range(no_iterations):
			if (j%10) == 0:
				delete_set = []
				full_B = np.zeros((len(B), 1))
				full_B[relevant_variables] = reduced_B
				theta = dual_feasible_scaler_modified(X, full_B, Y, L, Residual, reduced_X)
				c = theta
				r_big = L2(theta - Y / L)
				r_small = (1 / L) * math.sqrt(max((1 - L2(Residual)**2 - 2*L*L1(reduced_B)), 0))
				r = math.sqrt(abs(r_big**2 - r_small**2))
				for k in relevant_variables:
					if abs(np.dot(X[:,k], c)) + r < 1:
						irrelevant_variables.append(k)
						delete_set.append(k)
				for x in delete_set:
					relevant_variables.remove(x)
				reduced_X = np.delete(X, irrelevant_variables, 1)
				reduced_B = np.delete(full_B, irrelevant_variables, 0)
				if len(reduced_X[0]) == 0:
					break
				if duality_gap_modified(reduced_X, reduced_B, Y, L, Residual, reduced_X) < universal_duality_gap:
					break 
			for k in range(len(reduced_B)):
				prev = copy.copy(reduced_B[k])
				reduced_B = coordinate_update(reduced_X, reduced_B, Y, L, k, Residual)
				Residual += reduced_X[:,k]*(prev - reduced_B[k])
		total_irrelevant_variables.append(len(irrelevant_variables))
		B = np.zeros((p, 1))
		B[relevant_variables] = reduced_B
	return(total_irrelevant_variables, total_times)


#Working Set Implementation-> Based on implementation sketched out by Massias et al. -> Solve the Lasso problem over a restricted set of features selected based on correlation with residual

def gap_safe_rule_working_set(X, B, Y, Lambdas):
	total_times = []
	p = len(B)
	n = len(X)
	total_irrelevant_variables = []
	theta = np.zeros((n, 1))
	for L in Lambdas:
		time_start = time.perf_counter()
		unsafely_reduced_X = X
		unsafely_reduced_B = B
		relevant_variables = [x for x in range(len(B))]
		Residual = Y - np.matmul(X, B)
		Residual = Residual.reshape(-1)
		E = Y / L
		non_zero_count = 0
		previously_active = []
		for t in range(no_iterations):
			alpha = []
			for j in range(p):
				X_j = np.transpose(X[:,j].reshape((n,1)))
				if abs(np.dot(X_j, E)) <= 1.0000001:
					alpha.append(1)
				else:
					alpha.append((1 - abs(np.dot(X_j, theta))) / abs(np.dot(X_j, (E - theta))))
			alpha = min(alpha)
			theta = (1 - alpha)*theta + alpha*E
			d_gap = compute_lasso_error(X, B, Y, L) - (1/2) + (L**2 / 2)*((L2(theta - (Y / L)))**2)
			if d_gap < universal_duality_gap:
				if t == 0:
					total_irrelevant_variables.append(p - 1)
				else:
					total_irrelevant_variables.append(p - np.count_nonzero(B))
				break
			delete_set = []
			r_big = L2(theta - Y / L)
			r_small = (1 / L) * math.sqrt(max((1 - L2(Residual)**2 - 2*L*L1(unsafely_reduced_B)), 0))
			r = math.sqrt(abs(r_big**2 - r_small**2))
			temp = []
			for k in relevant_variables:
				corr = abs(np.dot(X[:,k], theta))
				if corr + r < 1:
					delete_set.append(k)
				else:
					if k in previously_active:
						temp.append((k, 1))
					else:
						temp.append((k, corr))
			for x in delete_set:
				relevant_variables.remove(x)
			temp.sort(key=lambda x: x[1], reverse = True)
			ws_size = max(50, min(2*non_zero_count, p))
			working_set = []
			for x in range(min(ws_size, len(temp))):
				working_set.append(temp[x][0])
			unsafely_reduced_X = X[:, working_set]
			unsafely_reduced_B = B[working_set]
			unsafely_reduced_B, E = coordinate_descent_working_set(unsafely_reduced_X, unsafely_reduced_B, Y, L, theta)
			non_zero_count = np.count_nonzero(unsafely_reduced_B)
			non_zero_indices = np.nonzero(unsafely_reduced_B)[0].tolist()
			previously_active = list(working_set[i] for i in non_zero_indices)
			B[working_set] = unsafely_reduced_B
			Residual = Y - np.matmul(unsafely_reduced_X, unsafely_reduced_B)
			Residual = Residual.reshape(-1)
		time_elapsed = (time.perf_counter() - time_start)
		total_times.append(time_elapsed)
	return(total_irrelevant_variables, total_times)


def coordinate_descent_working_set(X, B, Y, Lambda, theta): 
	Residual = Y - np.matmul(X, B)
	Residual = Residual.reshape(-1)
	for i in range(no_iterations):
		for k in range(len(B)):
			prev = copy.copy(B[k])
			B = coordinate_update(X, B, Y, Lambda, k, Residual)
			Residual += X[:,k]*(prev - B[k])
		if (i % 10) == 0:
			E = dual_feasible_scaler(X, B, Y, Lambda)
			primal_value = compute_lasso_error(X, B, Y, Lambda)
			if len(X[0]) == 0:
				break
			if primal_value - (1/2) + (Lambda**2 / 2)*((L2(E - (Y / Lambda)))**2) < 0.3*(primal_value - (1/2) + (Lambda**2 / 2)*((L2(theta - (Y / Lambda)))**2)):
				break
	return(B, E)



