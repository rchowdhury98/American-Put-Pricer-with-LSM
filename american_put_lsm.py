import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
from sympy.solvers import solve
from sympy import Symbol

# Main Reference: https://www.cs.rpi.edu/~magdon/ -> Teaching -> Computational Finance -> Course Notes / Handouts -> Pricing the American Option

def get_predictor_from_paths(stock_price, paths, degree):
	prices = stock_price * np.exp(paths)
	
	price_matrix = prices
	
	for d in range(2,degree+1):
		price_matrix = np.hstack((price_matrix, prices**d))

	return price_matrix

if __name__ == "__main__":
	if len(sys.argv)!=9:
		print("Insufficient args!")
		quit()

	sigma = float(sys.argv[1])			# real world volatility
	stock_price = float(sys.argv[2])	# starting security price
	strike_price = float(sys.argv[3])	# strike price
	r = float(sys.argv[4])				# risk free return rate
	delta_t = float(sys.argv[5])		# interval for timesteps
	n = int(sys.argv[6])				# timesteps
	M = int(sys.argv[7])				# # of Monte Carlo paths to generate 
	degree = int(sys.argv[8])			# degree of polynomial to fit to

	T = n*delta_t						#Calculate termination time for the contract

	
	#First set of Monte Carlo generations to estimate the optimal exercise function

	#Generate M paths at the termination time (T)
	paths = np.random.normal((r-0.5*(sigma**2))*T, np.sqrt(T)*sigma, (M,1))

	#Initialize the optimal excercise funtion pi (at time T we can only exercise if the price is at least K)
	pi = np.ones((1,n+1)) * strike_price

	#Initialize the values vectior, at time T the value of the option the value of exercising
	v = np.fmax(np.zeros((paths.shape)), (strike_price - stock_price*np.exp(paths)))

	#Begin first set of loops
	for i in np.arange(n-1,0,-1):
		for j in np.arange(0,M):
			paths[j] = np.random.normal(paths[j]*(1-(1/(i+1))), sigma*np.sqrt(delta_t*(1-(1/(i+1)))))	#Backwards time step in Monte carlo
			v[j] = np.exp(-r*delta_t)*v[j]																#The values of the next time step discounted to the current value

		# Build the predictor matrix
		price_matrix = get_predictor_from_paths(stock_price, paths, degree)
		
		fit = linear_model.LinearRegression().fit(price_matrix,v)

		# if fit.intercept_[0] != 0 or (fit.coef_[0] != 0).any(): #temp sanity check

		# Use solver to determine where the estimated holding function equals the value of exercising ( fit.coef dot price_matrix = (K - S)+)
		# to do - look into tradeoff of using simple polynomials vs Laguerre and hermite polynomials
		x = Symbol('x', positive=True, real=True)
		coef_dot_x = '{intercept}'.format(intercept=fit.intercept_[0])
		for c, k in enumerate(fit.coef_[0,:]):
			if k < 0:
				coef_dot_x = coef_dot_x + ' {k} * x**{c}'.format(k=k, c=c+1)
			else:
				coef_dot_x = coef_dot_x + ' + {k} * x**{c}'.format(k=k, c=c+1)

		# find the lowest, real, positive solution to define the proper exercise boundary
		# to do for cubic + quadratic fits : this often returns no real results. Options for fixes:
		#	- use the previous boundary function (may lose accuracy - check papers on this, potential smooth using average?)
		#	- drop down to quadratic / linear

		# Need two potential solves to deal with (K-S)+, first for the case K-S > 0, and then for K - S = 0 when exercising is not valid
		solutions_plus = [s for s in solve(coef_dot_x + ' - ({k}-x)'.format(k=strike_price)) if s.is_real and s < strike_price and s > 0]

		if len(solutions_plus) != 0:
			pi[0,i] = solutions_plus[0]
		else:
			solutions_reg = [s for s in solve(coef_dot_x.format(k=strike_price)) if s.is_real and s < strike_price and s > 0]

			if len(solutions_reg) != 0:	
				pi[0,i] = solutions_reg[0]
			else: #use previous boundary function (see above todo)
				pi[0,i] = pi[0,i+1]

		# check along the paths for where it is optimal to exercise and set the value accordingly
		for j in np.arange(0,M):
			if stock_price * np.exp(paths[j]) <= float(pi[0,i]):
				v[j] = np.max([0,strike_price - (stock_price * np.exp(paths[j]))])

		if i%100 == 0:
			print("Steps left: ", i)
			print(np.average(v))

	# Here we run into the issue at timestep 0 where all the values will be equal to S
	# We avoid fitting a model here and take the average of the future time steps for the value at 0 and discount the value
	# Unlike below, calculating the value is unecessary, we just assume continuity and discount the exercise boundary
	pi[0,0] = np.exp(-r*delta_t)*pi[0,1]

	# Avoid positive bias on prices by carrying over the optimal exercise function from the previous monte carlo, and using it to calculate the value on a new set of paths
	paths = np.random.normal((r-0.5*(sigma**2))*T, np.sqrt(T)*sigma, (M,1))
	v = np.fmax(np.zeros((paths.shape)), (strike_price - stock_price*np.exp(paths)))


	for i in np.arange(n-1,0,-1):
		for j in np.arange(0,M):
			paths[j] = np.random.normal(paths[j]*(1-(1/(i+1))), sigma*np.sqrt(delta_t*(1-(1/(i+1)))))
			v[j] = np.exp(-r*delta_t)*v[j]

			if stock_price * np.exp(paths[j]) <= float(pi[0,i]):
				v[j] = np.max([0,strike_price - (stock_price * np.exp(paths[j]))])

		
		if i%100 == 0:
			print("Steps left: ", i)
			print(np.average(v))

	# Here we run into the issue at timestep 0 where all the values will be equal to S
	# We avoid fitting a model here and take the average of the future time steps for the value at 0 and discount the value
	v_h = np.average(v)
	v[0] = np.max([np.max([0, strike_price - stock_price]), v_h])
	print(v[0])
