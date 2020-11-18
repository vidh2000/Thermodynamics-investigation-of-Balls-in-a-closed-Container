"""
@author: Vid Homšak
"""
import numpy as np 
import random as rand
from Ball import *
from Simulation import *
import matplotlib.pyplot as plt
from PositionGenerator import *
from function_fits import *
"""
This file contains functions with self explanatory names for the functions from Thermodynamics Snookered - main.py.

Functions (called in Thermodynamics Snookered.py) use generate_positions() function
from PositionGenerator.py, to create arbitrary amount of balls inside the container.

__________________________________________________________________________________________________________________________
The functions below (automatically) adjust global variables from Simulation.py,
by inputting appropriate values as optional parameters, to only give out relevant output required by each function,
when it is called in Thermodynamics Snookered.py.
For understanding of what each variable means (i.e PLOTS, PHYSICS, PRESSURE etc.),
go to Simulation.py where they're explained on the top of the file in detail,
BUT users doesn't have to (shouldn't) change anything as constants' values are appropriately adjusted
in the function definitions below for them. Furthermore, this file should serve only as a reference for the understanding
of the functions used in Thermodynamics Snookered.py as user can (should) manipulate
all relevant values directly from the Thermodynamics Snookered.py
___________________________________________________________________________________________________________________________
"""

															    # ((1)) #

def simulation_animation_without_physics(radius_container, N_balls, alpha, v_initial, v_sigma, num_frames, dt_frame, N_data_points=20,
										 animate=1, ball_mass=1E-2):
	"""
	Graphical animation of the balls in the container simulation (no physics plots/data output).
	It sets the following variables in Simulation.py to == 1, others are == 0;
		PROGRESS = 1
	"""

	# Create a grid with initial positions of the N^2 balls
	radius_ball, positions = generate_positions(radius_container, N_balls, alpha)

	# Create the array of Ball objects
	balls = np.array([])
	for i in range(N_balls*N_balls):
		v 		= rand.normalvariate(v_initial, v_sigma)
		phi 	= rand.uniform(0,2*np.pi)
		balls 	= np.append(balls, Ball(mass = ball_mass, radius = radius_ball, position = positions[i],
										velocity = [v*np.cos(phi),v*np.sin(phi)], colour = "r"))
		
	# Simulation class initialised
	sim = Simulation(balls, N_data_points = N_data_points, radius_container = -radius_container, dt_frame=dt_frame, progress=1, plots=0,
					 physics=0, ball_distance_to_center=0, ball_relative_distances=0, speed_distribution=0,
					 velocity_components_distributions=0, average_speeds=0, average_velocity_components=0, temperature=0,
					 pressure=0, ke=0, momentum=0)
	print(f"Simulation animation of {N_balls*N_balls} balls with initial speed distribution: Norm({v_initial},{v_sigma}), mass m = {ball_mass} kg, radius r = %.2E m in a container with radius R = {float(-sim._container.radius())} m." % radius_ball)
	
	# Run the simulation
	sim.run(num_frames = num_frames, animate = animate, dt_frame = dt_frame)

															    # ((2)) #

def time_dependances_and_distributions(radius_container, N_balls, alpha, v_initial, v_sigma, N_data_points, num_frames, animate,
									   ball_mass=1E-2, dt_frame=1E-6):
	"""
	Get time dependent function plots and histograms.
			function(time): average_speed(time), average_velocity_component(time), temperature(time), pressure(time).
			Histograms: 	speed distribution, velocity components distributions,	
							relative distances between ball balls, distances of balls to the container center
	It sets the following variables in Simulation.py to == 1, others are == 0;
		PROGRESS = 1
		PLOTS 	 = 1
		PHYSICS  = 1
		BALL_DISTANCE_TO_CENTER             = 1
    	BALL_RELATIVE_DISTANCES             = 1
    	SPEED_DISTRIBUTION                  = 1
    	VELOCITY_COMPONENTS_DISTRIBUTIONS   = 1
    	AVERAGE_SPEEDS                      = 1
    	AVERAGE_VELOCITY_COMPONENTS         = 1
    	TEMPERATURE                         = 1
    	PRESSURE                            = 1
	"""

	# Create a grid with initial positions of the N^2 balls
	radius_ball, positions = generate_positions(radius_container, N_balls, alpha)

	# Create the array of Ball objects
	balls = np.array([])
	for i in range(N_balls*N_balls):
		v 		= rand.normalvariate(v_initial, v_sigma)
		phi 	= rand.uniform(0,2*np.pi)
		balls 	= np.append(balls, Ball(mass = ball_mass, radius = radius_ball, position = positions[i],
										velocity = [v*np.cos(phi),v*np.sin(phi)], colour = "r"))
		
	# Simulation class initialised
	sim = Simulation(balls, N_data_points = N_data_points, radius_container = -radius_container, dt_frame = dt_frame, progress=1,
					 plots=1, physics=1, ball_distance_to_center=0, ball_relative_distances=0, speed_distribution=1,
					 velocity_components_distributions=1, average_speeds=1, average_velocity_components=0, temperature=1,
					 pressure=0, ke=0, momentum=0)
	print(f"Simulation: {N_balls*N_balls} balls with initial speed distribution: Norm({v_initial},{v_sigma}), mass m = {ball_mass} kg, radius r = %.2E m in a container with radius R = {float(-sim._container.radius())} m." % radius_ball)
	
	# Run the simulation
	sim.run(num_frames = num_frames, animate = animate)




															    # ((3)) #

def pressure_vs_temperature(radius_container, N_balls, animate, radius_ball, v_initial, v_sigma, nb_different_radiuses,
							nb_variations, num_frames, ball_mass, dt_frame=1E-6):
	"""
	(3.A) section - p(T)
	This function runs the simulation nb_variations times, for nb_different_radiuses
	to output p(T) matplotlib.pyplot graph, where data is fitted by p_T_fit() function
	and the values for the parameter b from Van der Waals equation are found with the
	uncertainty calculated from the covariance matrix - see functions_fits.py
	It sets the following variables in Simulation.py to == 1, others are == 0;
		PHYSICS 	= 1
		if PHYSICS:
    		TEMPERATURE                         = 1
    		PRESSURE                            = 1
	"""
	for radius_var in range(1, nb_different_radiuses+1):
		# PV = nRT related data attributes
		pressure 	= np.array([])
		temperature = np.array([])
		volume 	 	= np.pi*radius_container*radius_container
		# Create a grid with initial positions of the N^2 balls
		radius_ball_max, positions = generate_positions(radius_container, N_balls, 0.99)
		# Check if the inputted argument radius_ball isn't too big so that balls aren't overlapping
		if radius_ball > radius_ball_max:
			raise ValueError(f"Designated radius_ball is too big for initialisation of {N_balls*N_balls} particles, make it smaller.")

		# Looping variations parameters (how much the varying quantities change in total)
		if radius_var < 5:
				alpha = 0.2
		elif (radius_var <=8 and radius_var >=5): 
			alpha = 0.5
		elif radius_var == 9:
			alpha = 1
		elif radius_var == 10:
			alpha = 3

		beta  = 10*v_initial/nb_variations

		# Different sets of initial particles
		for variation in range(1,nb_variations+1):

			# Create/renew the array of Ball objects
			balls = np.array([])
			for i in range(N_balls*N_balls):
				v 		= rand.normalvariate(v_initial+beta*variation, v_sigma)
				phi 	= rand.uniform(0,2*np.pi)
				balls 	= np.append(balls, Ball(mass = ball_mass, radius = radius_ball/radius_var**(alpha), position = positions[i],
									velocity = [v*np.cos(phi),v*np.sin(phi)], colour = "r"))
				
			# Simulation class initialised
			sim = Simulation(balls, N_data_points=5, radius_container = -radius_container, dt_frame=dt_frame, progress=0, plots=0, physics=1,
						 ball_distance_to_center=0, ball_relative_distances=0, speed_distribution=0, velocity_components_distributions=0,
	                 	 average_speeds=0, average_velocity_components=0, temperature=1, pressure=1, ke=0, momentum=0)
			print(f"p(T) simulation: {N_balls*N_balls} balls with initial speed distribution: Norm({v_initial*variation},{v_sigma}), mass m = {ball_mass} kg, radius r = %.2E m in a container with radius R = {float(-sim._container.radius())} m.  (Progress: ({radius_var}/{nb_different_radiuses}) {variation}/{nb_variations})" % sim._balls[0].radius())
			
			# Run the simulation
			sim.run(num_frames = num_frames, animate = animate)
			temperature 	= np.append(temperature, sim.get_average_temperature())
			pressure 		= np.append(pressure, sim.get_average_pressure())
  
		#Plotting for each radius. Best fit line found by p_T_fit() function from function_fits.py
		pressure_fit, parameters_fit, uncertainty = p_T_fit([np.pi*radius_ball**2], temperature, pressure, N_balls**2, volume)
		# Van der Waals equation
		b 		= parameters_fit[0]
		b_std 	= uncertainty[0]
		plt.scatter(temperature, pressure)
		plt.plot(temperature, pressure_fit, label = f"r=%.2E m, b=(%.2E $\pm$ %0.1E) m$^2$"
													% (radius_ball/radius_var**(alpha), b, b_std))
		#print(f"Parameter b in Van der Waals equation, b = {b}, which is {b/(np.pi*radius_ball**2)}x the volume of a ball")
	
	# Plotting
	params = {
		'axes.labelsize':24,
		'font.size': 20,
		'legend.fontsize': 7,
		'xtick.labelsize': 16,
		'ytick.labelsize': 16,
		'figure.figsize': [8,8/1.618],
	}
	plt.rcParams.update(params)
	plt.grid()
	plt.xlabel("Temperature [K]")
	plt.ylabel("Pressure [Pa]")
	# Equation of state - PV = Nk_BT plot
	pressure_from_equation_of_state_T = equation_of_state_p_T(temperature, N_balls**2, volume)
	plt.plot(temperature, pressure_from_equation_of_state_T, c='r', label = "Equation of state", linewidth=3.0)
	plt.legend()
	plt.show()

def pressure_vs_volume(radius_container, N_balls, radius_ball, animate, v_initial, v_sigma, nb_variations, num_frames,
					   nb_different_radiuses, ball_mass=1E-4, dt_frame=1E-6):
	"""
	(3.B) section - p(V)
	This function runs the simulation nb_variations times, for nb_different_radiuses
	to output p(V) matplotlib.pyplot graph, where data is fitted by p_V_fit() function
	and the values for the parameter b from Van der Waals equation are found with the
	uncertainty calculated from the covariance matrix - see functions_fits.py
	It sets the following variables in Simulation.py to == 1, others are == 0;
		PHYSICS 	= 1
		if PHYSICS:
    		PRESSURE 	= 1
    		TEMPERATURE = 1
	"""

	for radius_var in range(1, nb_different_radiuses+1):
		# PV = nRT related data attributes
		pressure 	= np.array([])
		volume 		= np.array([])
		temperature = 0

		# Create a grid with initial positions of the N^2 balls
		radius_ball_max, positions = generate_positions(radius_container, N_balls, 0.99)
		# Check if the inputted argument radius_ball isn't too big so that balls aren't overlapping
		if radius_ball > radius_ball_max:
			raise ValueError(f"Designated radius_ball is too big for initialisation of {N_balls*N_balls} particles, make it smaller.")
		
		# Looping variations parameters (how much the varying quantities change in total)
		if radius_var < 5:
				alpha = 0.2
		elif (radius_var <=8 and radius_var >=5): 
			alpha = 0.5
		elif radius_var == 9:
			alpha = 1
		elif radius_var == 10:
			alpha = 3
		beta = 3*radius_container/nb_variations

		# Different sets of initial particles
		for variation in range(1,nb_variations+1):

			# Create/renew the array of Ball objects
			balls = np.array([])
			for i in range(N_balls*N_balls):
				v 		= rand.normalvariate(v_initial, v_sigma)
				phi 	= rand.uniform(0,2*np.pi)
				balls 	= np.append(balls, Ball(mass = ball_mass, radius = radius_ball/radius_var**(alpha), position = positions[i],
												velocity = [v*np.cos(phi),v*np.sin(phi)], colour = "r"))
				
			# Simulation class initialised
			sim = Simulation(balls, N_data_points=5, radius_container = -variation*beta-radius_container, dt_frame=dt_frame, progress=0,
							 plots=0, physics=1, ball_distance_to_center=0, ball_relative_distances=0, speed_distribution=0,
							 velocity_components_distributions=0, average_speeds=0, average_velocity_components=0, temperature=1,
							 pressure=1, ke=0, momentum=0)
			print(f"p(V) simulation: {N_balls*N_balls} balls with initial speed distribution: Norm({v_initial},{v_sigma}), mass m = {ball_mass} kg, radius r = %.2E m in a container with radius R = %.2E m.  (Progress: ({radius_var}/{nb_different_radiuses}) {variation}/{nb_variations})" % (radius_ball,-sim._container.radius()))
			
			# Run the simulation
			sim.run(num_frames = num_frames, animate = animate)
			volume 			= np.append(volume, np.pi*(sim._container.radius())**2)
			pressure 		= np.append(pressure, sim.get_average_pressure())
			temperature 	= sim.get_average_temperature()

		# Plotting. Best fit line found by p_V_fit() function from function_fits.py
		pressure_fit, parameters_fit, uncertainty = p_V_fit([np.pi*radius_ball**2], volume, pressure, N_balls**2, temperature)
		# Van der Waals equation
		b 		= parameters_fit[0]
		b_std 	= uncertainty[0]
		plt.scatter(volume, pressure)
		plt.plot(volume, pressure_fit, label = f"r = %.2E m, b = (%.2E $\pm$ %0.1E) m$^2$"
											% (radius_ball/radius_var**(alpha), b, b_std))
		#print(f"Parameter b in Van der Waals equation, b = {b}, while the volume of a ball, V = {np.pi*radius_ball**2}")

	# Plotting.
	params = {
		'axes.labelsize':24,
		'font.size': 20,
		'legend.fontsize': 7,
		'xtick.labelsize': 16,
		'ytick.labelsize': 16,
		'figure.figsize': [8,8/1.618],
	}
	plt.rcParams.update(params)
	plt.grid()
	plt.xlabel("Volume [m$^2$]")
	plt.ylabel("Pressure [Pa]")
	# Equation of state - PV = Nk_BT plot
	pressure_from_equation_of_state_V = equation_of_state_p_V(volume, N_balls**2, temperature)
	plt.plot(volume, pressure_from_equation_of_state_V, c='r', label = "Equation of state", linewidth=3.0)
	plt.legend()
	plt.show()


def pressure_vs_number_of_balls(radius_container, N_smallest, N_biggest, radius_ball, animate, v_initial, v_sigma, num_frames,
								nb_different_radiuses, ball_mass, dt_frame=1E-6):
	"""
	(3.C) section - p(N)
	This function runs the simulation (N_bigest-N_smallest) times, for nb_different_radiuses
	to output p(N) matplotlib.pyplot graph, where data is fitted by p_N_fit() function
	and the values for the parameter b from Van der Waals equation are found with the
	uncertainty calculated from the covariance matrix - see functions_fits.py
	It sets the following variables in Simulation.py to == 1, others are == 0;
		PHYSICS 	= 1
		if PHYSICS:
    		PRESSURE 	= 1
    		TEMPERATURE = 1
	"""
	for radius_var in range(1, nb_different_radiuses+1):
		# PV = nRT related data attributes
		pressure 		= np.array([])
		N_balls_arr 	= np.array([N for N in range(N_smallest, N_biggest+1)])
		volume 		= np.pi*radius_container**2
		temperature = 0

		# Check if the inputted argument radius_ball isn't too big so that balls aren't overlapping
		radius_ball_max = generate_positions(radius_container, N_biggest, 0.95)[0]
		if radius_ball > radius_ball_max:
			raise ValueError(f"Designated radius_ball is too big for initialisation of {N_biggest*N_biggest} particles, make it smaller.")

		# Looping variations parameters (how much the varying quantities change in total)
		if radius_var < 5:
				alpha = 0.2
		elif (radius_var <=8 and radius_var >=5): 
			alpha = 0.5
		elif radius_var == 9:
			alpha = 1
		elif radius_var == 10:
			alpha = 5

		# Different sets of initial particles
		for N in N_balls_arr:
			# Create/renew the array of Ball objects
			balls = np.array([])
		
			# Create a grid with initial positions of the N^2 balls
			positions = generate_positions(radius_container, N, 0.99)[1]
			
			for i in range(N*N):
				v 		= rand.normalvariate(v_initial, v_sigma)
				phi 	= rand.uniform(0,2*np.pi)
				balls 	= np.append(balls, Ball(mass = ball_mass, radius = radius_ball/radius_var**(alpha),
												position = positions[i], velocity = [v*np.cos(phi),v*np.sin(phi)], colour = "r"))
				
			# Simulation class initialised
			sim = Simulation(balls, N_data_points=10, radius_container = -radius_container, dt_frame=dt_frame, progress=0, plots=0,
							physics=1, ball_distance_to_center=0, ball_relative_distances=0, speed_distribution=0,
							velocity_components_distributions=0, average_speeds=0, average_velocity_components=0, temperature=1,
							pressure=1, ke=0, momentum=0)
			print(f"p(N) simulation: {N*N} balls with initial speed distribution: Norm({v_initial},{v_sigma}), mass m = {ball_mass} kg, radius r = %.2E m in a container with radius R = {float(-sim._container.radius())} m.  (Progress: ({radius_var}/{nb_different_radiuses}) {N-N_smallest+1}/{N_biggest+1-N_smallest})" % radius_ball)
			
			# Run the simulation
			sim.run(num_frames = num_frames, animate = animate)
			pressure 		= np.append(pressure, sim.get_average_pressure())
			temperature 	= sim.get_average_temperature()

		# Plotting. Best fit line found by p_N_fit() function from function_fits.py
		pressure_fit, parameters_fit, uncertainty = p_N_fit([np.pi*radius_ball**2], N_balls_arr*N_balls_arr, pressure, temperature, volume)
		# Van der Waals equation
		b 		= parameters_fit[0]
		b_std 	= uncertainty[0]
		plt.scatter(N_balls_arr*N_balls_arr, pressure)
		plt.plot(N_balls_arr*N_balls_arr, pressure_fit, label = f"r = %.2E m, b = (%.2E $\pm$ %0.1E) m$^2$"
																% (radius_ball/radius_var**(alpha), b, b_std))
		#print(f"Parameter b in Van der Waals equation, b = {b}, while the volume of a ball, V = {np.pi*radius_ball**2}")
	
	# Plotting
	params = {
		'axes.labelsize':24,
		'font.size': 20,
		'legend.fontsize': 7,
		'xtick.labelsize': 16,
		'ytick.labelsize': 16,
		'figure.figsize': [8,8/1.618],
	}
	plt.rcParams.update(params)
	plt.grid()
	plt.xlabel("Number of balls")
	plt.ylabel("Pressure [Pa]")
	# Equation of state - PV = Nk_BT plot
	pressure_from_equation_of_state_N = equation_of_state_p_N(N_balls_arr*N_balls_arr, temperature, volume)
	plt.plot(N_balls_arr*N_balls_arr, pressure_from_equation_of_state_N, c='r', label = "Equation of state", linewidth=3.0)
	plt.legend()
	plt.show()