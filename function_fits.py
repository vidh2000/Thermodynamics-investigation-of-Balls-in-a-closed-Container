"""
@author: Vid Homšak
"""
from scipy.optimize import curve_fit
import numpy as np
"""
This file contains functions that are used for fitting data on plots in
Simulation_types.py and Simulation.py files
"""
#PHYSICS CONSTANTS
k_B         = 1.38065E-23

def equation_of_state_p_T(temperature_arr, N, volume):
	"""
	This function outputs a pressure array corresponding to the temperature_arr,
	which is based on the ideal gas assumption i.e using the equation of state (pV=NK_bT)
	"""
	pressure_arr = np.array(list(map(lambda T: N*k_B/volume*T, temperature_arr)))
	return pressure_arr

def equation_of_state_p_V(volume_arr, N, temperature):
	"""
	This function outputs a pressure array corresponding to the volume_arr,
	which is based on the ideal gas assumption i.e using the equation of state (pV=NK_bT)
	"""
	pressure_arr = np.array(list(map(lambda V: N*k_B*temperature/V, volume_arr)))
	return pressure_arr

def equation_of_state_p_N(N_balls_arr, temperature, volume):
	"""
	This function outputs a pressure array corresponding to the N_balls_arr, which is based on
	the ideal gas assumtion i.e using the equation of state (pV=Nk_BT)
	"""
	pressure_arr = np.array(list(map(lambda N: k_B*temperature/volume*N, N_balls_arr)))
	return pressure_arr

def p_T_fit(guesses, temperature_arr, pressure_arr, N, volume):
	"""
	This function intakes 2 arrays used as independent and dependent variables
	(temperature_arr and pressure_arr respectively).
	N 		: number of balls in the container,
	volume  : volume of the container
	Will be used in pressure_vs_temperature to fit on the data and show if
	p(T) is indeed linear relation.
	Guesses is a list of linear_function() parameter estimations used for fitting
	Outputs:
		fitting array, array containing the value of the fit parameter and its uncertainty
	"""

	def linear_function(x, b):
		return N*k_B/(volume-N*b)*x

	fit 	 = curve_fit(linear_function, temperature_arr, pressure_arr, p0 = guesses)
	#print('The pressure fit parameters are:', fit[0])
	data_fit = linear_function(temperature_arr, *fit[0])
	return data_fit, fit[0], np.sqrt(np.diag(fit[1]))

def p_V_fit(guesses, volume_arr, pressure_arr, N, T):
	"""
	This function intakes 2 arrays used as independent and dependent variables
	(volume_arr and pressure_arr respectively).
	N... number of balls in the container, T... temperature in the container
	Will be used in pressure_vs_volume to fit on the data and show if
	p(V) is indeed an inverse relation.
	Guesses is a list of linear_function() parameter estimations used for fitting
	Outputs:
		fitting array, array containing the value of the fit parameter and its uncertainty
	"""
	def inverse_function(x, b):
		return N*k_B*T/(x-N*b)

	fit 	 = curve_fit(inverse_function, volume_arr, pressure_arr, p0 = guesses)
	#print('The pressure fit parameters are:', fit[0])
	data_fit = inverse_function(volume_arr, *fit[0])
	return data_fit, fit[0], np.sqrt(np.diag(fit[1]))

def p_N_fit(guesses, N_balls_arr, pressure_arr, T, volume):
	"""
	This function intakes 2 arrays used as independent and dependent variables
	(N_balls_arr and pressure_arr respectively).
	T  			: temperature in the container
	volume  	: volume of the container
	Will be used in pressure_vs_nb_balls to fit on the data and show
	if p(N) is indeed linear relation.
	Guesses is a list of linear_function() parameter estimations used for fitting
	Outputs:
		fitting array, array containing the value of the fit parameter and its uncertainty
	"""
	def linear_function(x, b):
		return k_B*T/(volume-x*b)*x

	fit 	 = curve_fit(linear_function, N_balls_arr, pressure_arr, p0 = guesses)
	#print('The pressure fit parameters are:', fit[0])
	data_fit = linear_function(N_balls_arr, *fit[0])
	return data_fit, fit[0], np.sqrt(np.diag(fit[1]))

def Maxwell_Boltzmann_distribution_fit(guesses, speeds_arr, N_balls_histogram, mass, temperature):
	"""
	Used in Simulation.py to fit the speed distribution of the balls in the container.
	Outputs:
		fitting array, value of the fit parameter and its uncertainty
	"""
	def Max_Boltz_distrib(v, A):
		return A*v*np.exp(-(mass*v*v/(2*k_B*temperature)))

	fit 	 = curve_fit(Max_Boltz_distrib, speeds_arr, N_balls_histogram, p0 = guesses)
	data_fit = Max_Boltz_distrib(speeds_arr, *fit[0])
	return list(data_fit), fit[0][0], np.sqrt(np.diag(fit[1]))[0]

def Maxwellian_velocity_distribution_fit(guesses, velocity_component_arr, N_balls_histogram, mass, temperature):
	"""
	Used in Simulation.py to fit the velocity components' v_x and v_y distributions,
	of the balls in the container. Outputs:
		fitting array, value of the fit parameter and its uncertainty
	"""
	def Gaussian(v, A, mu):
		return A*np.exp(-(mass*(v-mu)**2/(2*k_B*temperature)))

	fit 	 = curve_fit(Gaussian, velocity_component_arr, N_balls_histogram, p0 = guesses)
	data_fit = Gaussian(velocity_component_arr, *fit[0])
	return data_fit, fit[0][1], np.sqrt(np.diag(fit[1]))[1]