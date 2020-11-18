"""
@author: Vid Homšak
"""
from Simulation_types import *

"""
This file is the "main.py" type file and is where Simulation.py, Ball.py, PositionGenerator.py,
Simulation_types.py, function_fits.py files are used as modules so that the simulation
of the balls in a closed container can be run and testing of the gas laws
in the simulation can be executed.
________________________________________________________________________________________________
The following libraries need to be installed on the computer for the simulation to work:
numpy, random, matplotlib (matplotlib.pyplot and matplotlib.pylab),
scipy (scipy.optimize), statistics, itertools.
________________________________________________________________________________________________

All input parameters have been chosen such that to give good results (and appropriate plots for
what the function wants) in relatively short time, so no need to change them. Everything can be
adjusted otherwise, but the functions are set to give optimal wanted results without need of
interfering.

All the information about the functions called below can be found in the file Simulation_types.py


TIP: Very irregularly, but it can happen that balls overlap during the simulation and ValueError
is raised.
Just rerun the simulation, most likely it will finish the job the second time.

__________________________________________________________________________________________________
INPUTS (alphabetically ordered). Note thatt not all parameters appear in every function:
	animate 			  : if ==1, then graphical animation of the simulation will appear.
	alpha 				  : ratio of ball diameter versus the size of the lattice cell.
						    To get a better feeling for how balls are initialised set alpha = 0.95
						    (seemingly they won't move as they have random velocities and will only
						    collide amongst themselves for a while. Set alpha back to something
						    smaller for a "normal" simulation)
	ball_mass 			  : mass of the ball(s)
	dt_frame  			  : the actual time period between when each frame as experienced by the user
	radius_ball  		  : radius of the balls/initial radius of the balls, that gets smaller
							through loops, if nb_different_radiuses > 1
	radius_container 	  : radius of the container in meters
	N_balls 			  : the simulation will contain N_balls^2 balls. i.e N_balls=7 --> 49 balls
							in the container
	N_smallest(=N_balls)  : in (3.C) for the function = pressure(N_balls), represents the smallest
							value in the independent variable N_balls_arr
	N_biggest(=N_balls)   : in (3.C) for the function = pressure(N_balls), represents the biggest
							value in the independent variable N_balls_arr
	N_data_points		  : how many (averaged) data points will be taken for physical quantities
							during the simulation;
						    on how many time intervals is the total number of frames separated
						    (see Simulation_types.time_dependances_and_distributions() and variables
						    descriptions in Simulation.py for a deeper insight)
	nb_variations 		  : in section ((3)), it represents the number of data points per radius
							used in plots for displaying PV=Nk_BT relations
	nb_different_radiuses : in ((3)) it represents the number of different radii used for doing the
							P(T/V/N) plots.
							It makes the simulation last nb_different_radiuses times longer
	num_frames 			  : total number of frames each simulation will run through
	v_initial 			  : mean of the speed distribution (Normal distribution)
	v_sigma 			  : variance of the speed distribution (Normal distribution)
____________________________________________________________________________________________________
"""




"""
													((1))
						Graphical animation of the simulation of the balls in the container
										(no physics plots/physics data output).
"""

simulation_animation_without_physics(radius_container=1, N_balls=4, alpha=0.3,
					 				 v_initial=100, v_sigma=30, num_frames=200, dt_frame=1E-6)





"""
													((2))
				Displaying time dependent graphs and distributions of a single simulation run.
					Outputs:
						Functions(time): average_speed(time), average_velocity_component(time),
										 temperature(time), pressure(time).
						Histograms: speed distribution, velocity components distributions,	
									relative distances between ball balls,
									distances of balls to the container center
"""

time_dependances_and_distributions(	radius_container=1, N_balls=7, animate=0,
								  	alpha=0.2, v_initial=100, v_sigma=30,
 								  	N_data_points = 50, num_frames=20000)

										



"""
													((3))
		Equation of state testing; p(V-Nb) = Nk_BT testing - Van der Waals equation is used to fit
		numerical data for various radii of the balls and also a line from the equation of state is
		displayed, so that user can compare how "ideal" the "gas" in our container actually is
"""

# (3.A) Checking the pressure(temperature). Output is a plot p(T) with a linear best fit line
# 				and the values for the parameter b from Van der Waals eq. with standard error
pressure_vs_temperature(		radius_container=1, N_balls=4, animate=0,
								radius_ball=0.12, ball_mass=1E-2, v_initial=100, v_sigma=3,
								nb_different_radiuses=4, nb_variations=50, num_frames=100)
 											
# (3.B) Checking pressure(volume). Output is a plot p(V) with an inverse function best fit line
#				and the values for the parameter b from Van der Waals eq. with standard error
pressure_vs_volume(				radius_container=1, N_balls=4, animate=0,
								radius_ball=0.12, ball_mass=1E-4, v_initial=100, v_sigma=3,
								nb_different_radiuses=4, nb_variations=30, num_frames=100)
												

#(3.C) 	Checking pressure(Number of balls in the container). Output is a plot p(N) with a linear
#	best fit line(s) and the values for the parameter b from Van der Waals eq. with standard error
pressure_vs_number_of_balls(	radius_container=1, N_smallest=2, N_biggest=8, animate=0,
								radius_ball=0.03, ball_mass=1E-2, v_initial=100, v_sigma=0.1,
								nb_different_radiuses=4, num_frames=400)
										        