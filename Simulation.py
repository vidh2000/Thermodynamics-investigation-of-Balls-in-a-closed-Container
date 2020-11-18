"""
@author: Vid Homšak
"""
from Ball import *
import matplotlib.pylab as pl
import numpy as np
import statistics as stats
import random as rand
from itertools import combinations 
import matplotlib.pyplot as plt
from function_fits import *

#PHYSICS CONSTANTS
k_B         = 1.38065E-23

"""
Simulation.py file computationally describes the simulation of the balls in a container.

As the project is very broad, and the task requests various different outputs,
not all graphs/printed statements are always wanted to be outputted.
Hence, I've created global constants that are adjusted in Simulation_types.py file
as the already given optional parameters to give the appropriate, needed output
when a certain function is called in the Thermodynamics Snookered.py.

Descriptions of what "CONSTANT"==1 causes, when Simulation class is initialised in Simulation_types.py file:

    PROGRESS    == 1:
    Each loop, the statement
    "Frame: current_frame/total_number_of_frames. Total time passed in the simulation: total_time [s]"
    is printed, letting you know how many frames are left until the end of the simulation and telling
    how much time has passed in the simulation reference frame.
    You can turn that OFF, but I personally hate not knowing how far the simulation is, if it is long.
    
    DEBUG       == 1: (that one can only be adjusted here, in the Simulation.py)
    Each loop bunch of statements are printed out, showing the values stored in certain arrays,
    values of some attributes, local variable values.. which were/can be useful, for debugging/checking
    in the case something doesn't work or gives weird, unexpected results.
    
    PLOTS       == 1:
    After the end of the simulation, plots showing time dependence and histograms of enabled physics quantities.
    For PLOTS to give out actual physics data, you MUST set PHYSICS == 1 as well. 

    PHYSICS     == 1:
    Will store the needed data and perform needed operations during the simulation, to output plots of
    various physical quantities as a function of time and plot various speed/velocity distributions
    and store data for checking the gas equation.
    Kinetic and momentum conservation are checked regardless of the PHYSICS variable value.
    WARNING -- can slow down the code significantly.
    PHYSICS is "further branched", so if you could further adjust physics data output if not all plots/data is needed:
        
        BALL_DISTANCE_TO_CENTER             == 1
        Stores data and later display a histogram of balls' distance from the central position.
        Data stored for all balls after every (total_number_of_frames/N_data_points) frames.

        BALL_RELATIVE_DISTANCES             == 1
        Stores relative distance between each pair of balls to be displayed as a histogram.
        Data stored for all ball pairs after every (total_number_of_frames/N_data_points) frames.

        SPEED_DISTRIBUTION                  == 1
        Stores speeds (velocity magnitudes) of all balls to be later plotted as a histogram,
        which should resemble Maxwell-Boltzmann distribution, which is fitted. 
        Data stored for all balls after every (total_number_of_frames/N_data_points) frames.
        
        VELOCITY_COMPONENTS_DISTRIBUTIONS   == 1
        Stores velocity component values for all balls to be later plotted as a histogram
        (both components on the same plot), which should for both v_x and v_y look like
        Gaussian distribution centered at 0 and data is fitted by it.
        Data stored for all balls after every (total_number_of_frames/N_data_points) frames.

        AVERAGE_SPEEDS                      == 1
        Stores data to display average speed of all the balls as a function of time.
        Speeds for all balls are averaged over the a time period corresponding to
        (total_number_of_frames/N_data_points) frames and stored as a value in the array.
        Hence, averaged (over each total_number_of_frames/N_data_points frames) speed values
        are used as data points to be plotted as a function of time.

        AVERAGE_VELOCITY_COMPONENTS         == 1
        Like for average speed, it will output a plot of average velocity component as a function of time.
        Velocity components' values are averaged over the time period corresponding to
        (total_number_of_frames/N_data_points) frames and stored as a value in the array.
        Hence, averaged (over each total_number_of_frames/N_data_points frames) velocity component values
        are used as data points to be plotted as a function of time.

        TEMPERATURE                         == 1
        Uses KE_total = f/2*N*k_B*T to find temperature in the container;
        f=2 as world in simulation is 2D, hence <KE_total> = N*k_B*T
        Plots T(t) and saves the average temperature for the current simulation configuration
        (number of balls, initial speeds...) to be used for pV=nRT investigation.
        Temperature values for each frame are averaged over the time period corresponding
        to the (total_number_of_frames/N_data_points) number of frames and stored as 1 value
        for each time period in an array. Hence, averaged (over each total_number_of_frames/N_data_points frames)
        temperature values are used as data points to be plotted as a function of time.
        
        PRESSURE                            == 1
        Uses F = p*A = p * 2*pi*radius_container = (change in momentum) / time period
        (corresponding to (total_number_of_frames/N_data_points) frames)
        to find pressure on the container walls. Plots p(t) and saves the average pressure
        for the current simulation configuration (number of balls, initial speeds...) to be used for
        pV=nRT investigation. Pressure values for each frame are averaged over the
        (total_number_of_frames/N_data_points) frames time period and stored as one value
        for each time period in an array. Hence, averaged (over each time period) pressure values
        are used as data points to be plotted as a function of time.

"""

DEBUG                               = 0
PROGRESS                                = 0  
PLOTS                                   = 0
PHYSICS                                 = 0
BALL_DISTANCE_TO_CENTER                 = 0
BALL_RELATIVE_DISTANCES                 = 0
SPEED_DISTRIBUTION                      = 0
VELOCITY_COMPONENTS_DISTRIBUTIONS       = 0
AVERAGE_SPEEDS                          = 0
AVERAGE_VELOCITY_COMPONENTS             = 0
TEMPERATURE                             = 0
PRESSURE                                = 0
KE                                      = 0
MOMENTUM                                = 0


"""
The block of code below describes the Simulation class.
For a USER, there is no need to scroll down, unless they want to change the code,
furthermore for a user, there was no necessity to even read this file as for all outputs you only need to
change parameters in Simulation_types.py.
All the outputs and very much most of the parameters can be (and are only meant to be)
changed in the file Simulation_types.py.

But if you are the assessor, then go ahead :)
"""

class Simulation():
    """
    Class Simulation is separated into sections for easier readability, each of which (where applicable)
    contains multiple methods, which are then all combined in one "main" method at the end of
    each section that utilizes the other ones:
    
    Sections:
    1. Self.attributes
    2. Useful functions
    3. Physics related methods and operations
    4. Collisions between particles
    5. Running the simulation
    """

    def __init__(self, balls, N_data_points, progress, plots, physics, ball_distance_to_center,
                 ball_relative_distances, speed_distribution, velocity_components_distributions,
                 average_speeds, average_velocity_components, temperature, pressure, ke, momentum,
                 dt_frame, pos_container = [0,0], radius_container = -10, colour = "b"):
        self._container                      = Ball(pos_container, [0,0], "b", 6E24, radius_container)
        self.__container_patch               = pl.Circle(self._container.pos(), abs(self._container.radius()),
                                                         ec='b', fill=False, ls='solid')
        self._balls                          = balls
        self._objects                        = balls
        self._objects                        = np.append(self._objects, self._container)
        # All possible combinations including container object and their respective indices
        self.__combinations, self.__combinations_indices    = self.ball_combinations(self._objects)
        # All possible combinations of balls and their respective indices
        self.__ball_pairs, self.__ball_pairs_indices        = self.ball_combinations(self._balls)
        
        # Physics data attributes             
        self._N_data_points                   = N_data_points
        self.__frame                          = 0
        self.__num_frames                     = 0
        self.__dt_frame                       = dt_frame
        self.__time                           = 0
        self.__timer                          = 0
        self.__time_arr                       = np.array([])
        self.__avg_speeds_of_frames           = np.array([])
        self.__avg_speed_arr                  = np.array([])
        self.__velocity_x_arr                 = np.array([])
        self.__velocity_y_arr                 = np.array([])
        self.__avg_velocity_x_of_frames       = np.array([])
        self.__avg_velocity_x_arr             = np.array([])
        self.__avg_velocity_y_of_frames       = np.array([])
        self.__avg_velocity_y_arr             = np.array([])
        self.__momentum_total_x               = 0
        self.__momentum_total_y               = 0
        for obj in self._objects:
            self.__momentum_total_x += obj.momentum_x()
            self.__momentum_total_y += obj.momentum_y()
        self.__momentum_total_x_arr           = np.array([])
        self.__momentum_total_y_arr           = np.array([])
        self.__area                           = 2*np.pi*abs(self._container.radius()) 
        self.__pressure                       = 0
        self.__pressure_arr                   = np.array([])
        self.__average_pressure               = 0
        self.__KE                             = 0
        for i in range(len(self._balls)):
            self.__KE                         += 1/2* self._balls[i].mass() * np.linalg.norm(self._balls[i].vel()) * np.linalg.norm(self._balls[i].vel())
        self.__KE_arr                         = np.array([])
        self.__temperature                    = 2/2*self.__KE/(len(self._balls)*k_B)
        self.__temperature_avg                = np.array([])
        self.__temperature_arr                = np.array([])
        self.__average_temperature            = 0

        # Extra, histograms/function plots related attributes
        self.__ball_distance_from_center_arr  = np.array([])
        self.__ball_relative_distance_arr     = np.array([])
        self.__speed_distribution_arr         = np.array([])

        # Global variables adjusted accordingly to the function calls in Simulation_types.py
        # (get the needed outputs and nothing more)
        # I know declaring variables global is bad practice, but since this is "backend" code
        # I just decided to leave it be, sorry:)
        global PROGRESS
        global PLOTS
        global PHYSICS
        global BALL_DISTANCE_TO_CENTER
        global BALL_RELATIVE_DISTANCES
        global SPEED_DISTRIBUTION
        global VELOCITY_COMPONENTS_DISTRIBUTIONS
        global AVERAGE_SPEEDS
        global AVERAGE_VELOCITY_COMPONENTS
        global TEMPERATURE
        global PRESSURE
        global KE
        global MOMENTUM
        PROGRESS    = progress
        PLOTS       = plots
        PHYSICS     = physics
        if PHYSICS:
            BALL_DISTANCE_TO_CENTER             = ball_distance_to_center
            BALL_RELATIVE_DISTANCES             = ball_relative_distances
            SPEED_DISTRIBUTION                  = speed_distribution
            VELOCITY_COMPONENTS_DISTRIBUTIONS   = velocity_components_distributions
            AVERAGE_SPEEDS                      = average_speeds
            AVERAGE_VELOCITY_COMPONENTS         = average_velocity_components
            TEMPERATURE                         = temperature
            PRESSURE                            = pressure
            KE                                  = ke
            MOMENTUM                            = momentum
        else:
            pass

    """
    NAME MANGLING 
    Functions for obtaining/setting mangled attributes.
    
    """

    def get_time(self):
        return self.__time

    def get_area(self):
        return self.__area

    def get_pressure(self):
        return self.__pressure
    
    def get_KE(self):
        return self.__KE
    
    def get_temperature(self):
        return self.__temperature

    def get_ball_pairs(self):
        return self.__ball_pairs

    def get_combinations(self):
        return self.__combinations

    def get_average_pressure(self):
        return self.__average_pressure

    def get_average_temperature(self):
        return self.__average_temperature


    """

    USEFUL FUNCTIONS
    __str__    Gives useful info about Simulation instances when print(instance).
    debug      If variable DEBUG = 1 above, you get useful information about what's happening
               during each frame (each collision) 
    
    """

    def __str__(self):
        container_str = f"Container: position = {self._container.pos()}, velocity = {self._container.vel}"
        balls_str     = ""
        for i in range(len(self._balls)):
            balls_str += f"\nBall_{i+1}: position = {self._balls[i].pos()}, velocity = {self._balls[i].vel}"
        return container_str + balls_str       

    def debug(self):
        if DEBUG:
            for ball in self._balls:
                print(ball)
            print("List of possible times with indices:", time_next_collision_list)
            print("Time until next collision:", time_next_collision)
            print("Collision index list:", time_next_collision_index_list)
            colided_pairs = "Colided pairs: "
            for index_pair in collision_pair_indices:
                colided_pairs += f"{index_pair} "
            print(colided_pairs)
            print("Total Time:", self.__time)
            print("Collided")
            for ball in self._balls:
                print(ball)
            print("\n")

    """

    PHYSICS
    The following functions deal with calculating/updating physical quantities such as pressure,
    temperature and kinetic energy...
    Function update_physics_parameters() includes all of them and will calculate the values of the quantities
    over (or at the end of) the time intervals corresponding to (total_number_of_frames/N_data_points) frames.  
    Also includes some other physics related functions and physical quantities tests/checks.
    
    """

    def ball_distance_to_center(self, N_bins=5, distance_range = None):
        """
        Finds and records how many balls can be found in each "distance" region/interval from the center of the container.
        Serves as function that gives out data for a later constructed histogram of the distance
        each ball extends from the central position.
        N = number of bins/intervals of distance from the center in which balls are then stored.
        Returns the needed arguments for the histogram to be plotted.
        """

        if PHYSICS:
            if BALL_DISTANCE_TO_CENTER:

                # To input self.attribute as an optional function argument I have to do the following:
                if distance_range is None:
                        distance_range = (0, -self._container.radius())

                distance      = lambda x,y : np.sqrt(x*x + y*y)
                
                # Appends all needed data to the list self.distance_from_center_data that will serve 
                # as arguments for histogram plotting in self.get_plots()
                if self.__frame > (self.__num_frames/self._N_data_points):
                    for ball in self._balls:
                        self.__ball_distance_from_center_arr = np.append(self.__ball_distance_from_center_arr,
                                                                         distance(ball.pos()[0],ball.pos()[1]))
                  
                return self.__ball_distance_from_center_arr

    def ball_relative_distance(self, N_bins = 10, distance_range = None):
        """
        Finds relative distances amongst all the balls and stores them in the array.
        Serves as a function which gives out data later used to construct the histogram of the
        distance between each pair of balls.
        Uses method self.ball_combinations() to get all possible pairs of the balls
        """
        if PHYSICS:
            if BALL_RELATIVE_DISTANCES:

                # To input self.attribute as an optional function argument I have to do the following:
                if distance_range is None:
                        distance_range = (0, -2*self._container.radius())

                # Quick lambda functions to be used later to find relative distance
                delta_x                                 = lambda b1, b2: abs(b1.pos()[0] - b2.pos()[0])
                delta_y                                 = lambda b1, b2: abs(b1.pos()[1] - b2.pos()[1])
                distance                                = lambda x,y : np.sqrt(x*x + y*y)

                if self.__frame > (self.__num_frames/self._N_data_points):
                    for pair in self.__ball_pairs:
                        x                                   = delta_x(pair[0], pair[1])
                        y                                   = delta_y(pair[0], pair[1]) 
                        relative_distance                   = distance(x,y)
                        self.__ball_relative_distance_arr   = np.append(self.__ball_relative_distance_arr, relative_distance)

                return self.__ball_relative_distance_arr


    def get_average_speed(self):
        """
        Finds average speed over time periods in the simulation.
        Fills up the array self.avg_speed_arr, which contains avg. speeds (floats) for each time period
        and is used later for plotting.
        """

        if PHYSICS:
            if AVERAGE_SPEEDS:

                avg_speed_frame                   = 0
                for ball in self._balls:
                    avg_speed_frame              += ball.speed()/len(self._balls)
                self.__avg_speeds_of_frames       = np.append(self.__avg_speeds_of_frames, avg_speed_frame)
                
                if self.__frame > (self.__num_frames/self._N_data_points):
                    avg_speed                     = stats.mean(self.__avg_speeds_of_frames)
                    self.__avg_speeds_of_frames   = np.array([])
                    self.__avg_speed_arr          = np.append(self.__avg_speed_arr, avg_speed)
    
    def get_speed_distribution(self):
        """
        Finds how many balls have a certain speed throughout the simulation.
        Returns data usable to plot a histogram later on (i.e should be Maxwell-Boltzmann distribution)
        """

        if PHYSICS:
            if SPEED_DISTRIBUTION:

                if self.__frame > (self.__num_frames/self._N_data_points):
                    for ball in self._balls:
                        self.__speed_distribution_arr   = np.append(self.__speed_distribution_arr, ball.speed())

                return self.__speed_distribution_arr

    def get_velocity_component_distributions(self):
        """
        Finds how many balls have certain value of each of the two velocity components v_x and v_y
        Returns data usable to plot a histogram later (v_x and v_y should be Gaussian distributions with avg = 0)
        """

        if PHYSICS:
            if VELOCITY_COMPONENTS_DISTRIBUTIONS:

                if self.__frame > (self.__num_frames/self._N_data_points):

                    for ball in self._balls:
                        self.__velocity_x_arr   = np.append(self.__velocity_x_arr, ball.vel()[0])
                        self.__velocity_y_arr   = np.append(self.__velocity_y_arr, ball.vel()[1])
                return self.__velocity_x_arr, self.__velocity_y_arr


    def get_average_velocity(self):
        """
        Averages velocity components of all balls over a time period and stores these values
        (avg. velocities of the time periods)
        in self.avg_velocity_comoponent_arr
        """

        if PHYSICS:
            if AVERAGE_VELOCITY_COMPONENTS:

                avg_velocity_x_frame               = 0
                avg_velocity_y_frame               = 0
                for ball in self._balls:
                        avg_velocity_x_frame       += ball.vel()[0]/len(self._balls)
                        avg_velocity_y_frame       += ball.vel()[1]/len(self._balls)
                self.__avg_velocity_x_of_frames      = np.append(self.__avg_velocity_x_of_frames, avg_velocity_x_frame)
                self.__avg_velocity_y_of_frames      = np.append(self.__avg_velocity_y_of_frames, avg_velocity_y_frame)

                if self.__frame > (self.__num_frames/self._N_data_points):
                    avg_velocity_x                 = stats.mean(self.__avg_velocity_x_of_frames)
                    avg_velocity_y                 = stats.mean(self.__avg_velocity_y_of_frames)
                    self.__avg_velocity_x_of_frames  = np.array([])
                    self.__avg_velocity_y_of_frames  = np.array([])
                    self.__avg_velocity_x_arr        = np.append(self.__avg_velocity_x_arr, avg_velocity_x)  
                    self.__avg_velocity_y_arr        = np.append(self.__avg_velocity_y_arr, avg_velocity_y)  



    def update_temperature(self):
        """
        For each time period it calculates the average temperature of the system based on
        KE_total = f/2*N*k_B*T=N*k_B*T relation;
        f=2 as 2D world
        """

        if PHYSICS:
            if TEMPERATURE:

                T = 2/2*self.__KE/(len(self._balls)*k_B)
                self.__temperature_avg          = np.append(self.__temperature_avg, T)
                if self.__frame > (self.__num_frames/self._N_data_points):
                    self.__temperature          = stats.mean(self.__temperature_avg)
                    self.__temperature_avg      = np.array([])
                    self.__temperature_arr      = np.append(self.__temperature_arr, self.__temperature)
                    self.__average_temperature  = stats.mean(self.__temperature_arr)
    
    def update_pressure(self):
        """
        Gives a pressure on the container by the bouncing walls.
        Found as 1/(2*pi*radius_container) * dp/dt where dp is total change in momentum
        of the balls during each time period.
        """

        if PHYSICS:
            if PRESSURE:

                if self.__frame > (self.__num_frames/self._N_data_points):
                    for ball in self._balls:
                        self.__pressure               += 1/self.__area * ball._momentum_change/self.__timer
                    self.__pressure_arr                = np.append(self.__pressure_arr, self.__pressure)
                
                    # Reset values
                    self._container.momentum_change    = 0
                    for ball in self._balls:
                        ball._momentum_change          = 0
                    self.__pressure                    = 0
                    self.__average_pressure            = stats.mean(self.__pressure_arr)

    def update_KE(self, KE_previous):
        """
        Updates kinetic energy and checks whether energy conservation was violated.
        Raises ValueError if it was.
        """

        self.__KE                   = 0
        for ball in self._balls:
            self.__KE               += 1/2* ball.mass() * np.linalg.norm(ball.vel()) * np.linalg.norm(ball.vel())
        if not abs(KE_previous - self.__KE) < 1E-3:
                raise ValueError(f"Kinetic energy wasn't conserved. KE_previous = {KE_previous}, KE_now = {self.__KE}, resulting in KE change = {self.__KE - KE_previous}")
        if PHYSICS:
            if KE:
                if self.__frame > (self.__num_frames/self._N_data_points):
                    self.__KE_arr                       = np.append(self.__KE_arr, self.__KE)


    def update_momentum(self, p_x_previous, p_y_previous):
        """
        Updates x and y momentum components and checks whether momentum conservation was violated.
        Raises ValueError if it was.
        """

        self.__momentum_total_x = 0
        self.__momentum_total_y = 0
        for obj in self._objects:
            self.__momentum_total_x += obj.momentum_x()
            self.__momentum_total_y += obj.momentum_y()

        if not abs(p_x_previous - self.__momentum_total_x) < 1E-3:
            raise ValueError(f"Momentum in x-direction wasn't conserved. p_x_previous = {p_x_previous}, p_x_now = {self.__momentum_total_x}, resulting in p_x change = {self.__momentum_total_x - p_x_previous}")

        if not abs(p_y_previous - self.__momentum_total_y) < 1E-3:
            raise ValueError(f"Momentum in y-direction wasn't conserved. p_y_previous = {p_y_previous}, p_y_now = {self.__momentum_total_y}, resulting in p_y change = {self.__momentum_total_y - p_y_previous}")
        
        if PHYSICS:
            if MOMENTUM:
                if self.__frame > (self.__num_frames/self._N_data_points):
                    self.__momentum_total_x_arr    = np.append(self.__momentum_total_x_arr, self.__momentum_total_x)
                    self.__momentum_total_y_arr    = np.append(self.__momentum_total_y_arr, self.__momentum_total_y) 

    def update_time(self):
        if PHYSICS:
            if self.__frame > (self.__num_frames/self._N_data_points):
                self.__time_arr                    = np.append(self.__time_arr, self.__time)
                self.__timer                       = 0
                self.__frame                       = 0

    def update_phyiscs_parameters(self, KE_previous, p_x_previous, p_y_previous):
        """
        Calls all the physics related functions defined above.
        """

        if BALL_DISTANCE_TO_CENTER:
            self.ball_distance_to_center()
        if BALL_RELATIVE_DISTANCES:
            self.ball_relative_distance()
        if SPEED_DISTRIBUTION:
            self.get_speed_distribution()
        if VELOCITY_COMPONENTS_DISTRIBUTIONS:
            self.get_velocity_component_distributions()
        if AVERAGE_SPEEDS:
            self.get_average_speed()
        if AVERAGE_VELOCITY_COMPONENTS:
            self.get_average_velocity()
        if TEMPERATURE:
            self.update_temperature()
        if PRESSURE:
            self.update_pressure()
        self.update_momentum(p_x_previous, p_y_previous)
        self.update_KE(KE_previous)
        self.update_time()


    """

    COLLISIONS BETWEEN PARTICLES
    This part utilizes Ball.collide() and Ball.time_to_collision() functions to find
    which pairs would collide (first), using itertools.combinations to sweep through all
    possibilities and then actually collides the pair(s) and also implements
    update_physics_parameters() to give out physical values.
    
    """

    def ball_combinations(self, objects_array):
        """
        Outputs 2 arrays:
        pairs =             contains all possible combinations of Ball type objects that can collide
        pairs_indices =     contains pairs' indices of the objects in pairs array as indexed in objects_array 
        """

        pairs                           = lambda array : np.array([pair for pair in combinations(array, 2)])
        pairs_indices                   = lambda array : np.array([indices for indices in combinations(range(len(array)), 2)])
        pairs                           = pairs(objects_array)
        pairs_indices                   = pairs_indices(objects_array)
        return pairs, pairs_indices


    def balls_intersecting_check(self, array_of_combinations):
        """
        Tests if balls (including edge of the container) are overlapping, which is physically impossible.
        Raises ValueError error saying how much they're overlapping if they do. 
        """

        for pair in array_of_combinations:
            separation  = np.sqrt(abs(pair[0].pos()[0] - pair[1].pos()[0])*abs(pair[0].pos()[0] - pair[1].pos()[0]) + abs(pair[0].pos()[1]
                            - pair[1].pos()[1])*abs(pair[0].pos()[1] - pair[1].pos()[1]))
            radius_sum  = pair[0].radius() + pair[1].radius()
            if separation < radius_sum:
                raise ValueError(f"Balls crossing boundaries by: {radius_sum - separation}")


    def find_collision_time_and_pair(self, combinations, combinations_indices):
        """
        Finds the smallest possible time out of all the possible collisions
        and tells which two objects will collide at this smallest time.
        Returns:
            time_next_collision     == float with value of the time of the next collision
            collision_pair_indices  == list containing indices of the pair that will collide
        """

        # time_next_collision_list of the form: [[time_i, combinations_indices_i], ...]
        time_next_collision_list        = []
        for i in range(len(combinations)):
            time_try                    = combinations[i][0].time_to_collision(combinations[i][1])
            if type(time_try) == np.float64:
                time_next_collision_list.append([time_try, combinations_indices[i]])

        # possible_next_collision_times of the form = [time_i, ...]
        possible_next_collision_times   = []
        for i in range(len(time_next_collision_list)):
            possible_next_collision_times.append(time_next_collision_list[i][0])

        time_next_collision             = min(possible_next_collision_times)       
        self.__time                     += time_next_collision
        self.__timer                    += time_next_collision

        time_next_collision_index_list  = [i for i, time in enumerate(possible_next_collision_times) if time == time_next_collision]
        collision_pair_indices          = []
        for index in time_next_collision_index_list:
            collision_pair_indices.append(time_next_collision_list[index][1])

        return time_next_collision, collision_pair_indices

    def collide_the_pair(self, collision_pair_indices):
        """
        Actually collide pair(s) that gave the shortest time until the next collision.
        If a ball has collided with the container wall, it adds momentum changes to allow
        a later calculation of the pressure in the system.
        """

        for index_pair in collision_pair_indices:
            obj1_momentum_change, obj2_momentum_change    = self._objects[index_pair][0].collide(self._objects[index_pair][1])

            # Add momentum changes to later find pressure
            if index_pair[0] == (len(self._objects)-1) or index_pair[1] == (len(self._objects)-1):
                #print("Collision with the wall")
                self._objects[index_pair][0]._momentum_change += obj1_momentum_change
                self._objects[index_pair][1]._momentum_change += obj2_momentum_change



    def next_collision(self):
        """
        Collides the appropriate pair of the Ball type objects
        and updates all balls' positions and physics attributes of the simulation.
        """

        # Store KE and momentum components before the collision as a local variable to later check
        # if energy/momentum conservation has been violated

        KE_previous                                 = self.__KE
        p_x_previous                                = self.__momentum_total_x
        p_y_previous                                = self.__momentum_total_y


        # Get time of next collision with indices of the specific pair that will collide.
        time_next_collision, collision_pair_indices = self.find_collision_time_and_pair(self.__combinations, self.__combinations_indices)

        # Move balls and the container (-1E-6 is here to take care of the overlapping bug without affecting realism of results)
        for obj in self._objects:
            obj.move(time_next_collision-1E-6)

        # Check if any objects intersect, which is physically impossible and raise an error if it happens.
        self.balls_intersecting_check(self.__combinations)
        
        # Collide the pair
        self.collide_the_pair(collision_pair_indices)

        # Update physical quantities and allow debugging
        self.update_phyiscs_parameters(KE_previous, p_x_previous, p_y_previous)
        self.debug()
    
    
    def get_plots(self):
        """
        Creates and displays matplotlib.pyplot graphs of the data needed/wanted.
        """ 

        if PLOTS:
            params = {
                    'axes.labelsize':18,
                    'font.size': 18,
                    'legend.fontsize': 11,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16,
                    'figure.figsize': [8.8,8.8/1.618],
                     }
    
            ### Histograms ###

            if BALL_DISTANCE_TO_CENTER:
                # Distance each ball extends from the central position
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Distance to center [m]")
                plt.ylabel("Number of balls")
                plt.hist(self.ball_distance_to_center(), 15)
                plt.show()

            if BALL_RELATIVE_DISTANCES:
                # Distance between each pair of balls
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Relative distance between ball pairs [m]")
                plt.ylabel("Number of ball pairs")
                plt.hist(self.ball_relative_distance(), 15)
                plt.show()

            if SPEED_DISTRIBUTION:
                # Speed distribution
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Speed [m/s]")
                plt.ylabel("Number of balls")
                bin_heights, bin_edges, patches     = plt.hist(self.__speed_distribution_arr, 30)
                bincentres                          = 0.5*(bin_edges[1:]+bin_edges[:-1])
                # Maxwell-Boltzmann fit and comparison to theoretical prediction
                speed_distribution_fit, A, A_unc              = Maxwell_Boltzmann_distribution_fit(
                                                        [800], bincentres, bin_heights,
                                                         self._balls[0].mass(), self.get_average_temperature())
                # Notable speeds
                v_avg_sim        = stats.mean(self.__avg_speed_arr)
                v_avg_theory     = np.sqrt(k_B*self.get_average_temperature()*np.pi/(2* self._balls[0].mass())) #was np.sqrt(8) x ...
                v_mp_sim         = bincentres[speed_distribution_fit.index(np.where(max(speed_distribution_fit),max(speed_distribution_fit),0))]
                v_mp_theory      = np.sqrt(k_B*self.get_average_temperature()/(self._balls[0].mass()))  #was * np.sqrt(2)
                print()
                print(self.__average_temperature)
                plt.plot(bincentres, speed_distribution_fit,
                label="Maxwell-Boltzmann distribution.\n$<v_{sim}>$ = %.2f ms$^{-1}$\n$<v_{theory}>$ = %.2f ms$^{-1}$\n$v_{mp_{sim}}$ = %.2f ms$^{-1}$\n$v_{mp_{theory}}$ = %.2f ms$^{-1}$"
                                                        % (v_avg_sim, v_avg_theory, v_mp_sim, v_mp_theory))
                plt.legend(loc = 1)
                plt.show()

            if VELOCITY_COMPONENTS_DISTRIBUTIONS:
                # Velocity distributions
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Velocity component magnitude [m/s]")
                plt.ylabel("Number of balls")
                bin_heights_x, bin_edges_x, patches_x = plt.hist(self.get_velocity_component_distributions()[0], 20,
                                                                 label = "$v_x$-component")
                bincentres_x                          = 0.5*(bin_edges_x[1:]+bin_edges_x[:-1])
                bin_heights_y, bin_edges_y, patches_y = plt.hist(self.get_velocity_component_distributions()[1], 20,
                                                                 label = "$v_y$-component")
                bincentres_y                          = 0.5*(bin_edges_y[1:]+bin_edges_y[:-1])
                v_x_fit, mu_x, mu_x_unc               = Maxwellian_velocity_distribution_fit(
                                                        [100,0], bincentres_x, bin_heights_x,
                                                        self._balls[0].mass(), self.get_average_temperature())
                v_y_fit, mu_y, mu_y_unc                = Maxwellian_velocity_distribution_fit(
                                                        [100,0], bincentres_y, bin_heights_y,
                                                        self._balls[0].mass(), self.get_average_temperature())
                plt.plot(bincentres_x, v_x_fit, label="$v_x$ Gaussian fit, $\mu$ = (%.1f $\pm$ %.1f) ms$^{-1}$" % (mu_x, mu_x_unc))
                plt.plot(bincentres_y, v_y_fit, label="$v_y$ Gaussian fit, $\mu$ = (%.1f $\pm$ %.1f) ms$^{-1}$" % (mu_y, mu_y_unc))
                plt.legend(loc=4)
                plt.show()

            ## Function plots over time ###
            
            if AVERAGE_SPEEDS:
                #Average speed(t)
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Average speed [m/s]")
                plt.axis([0,self.__time_arr[-1] + self.__time_arr[-1]- self.__time_arr[-2],0,1.5*self.__avg_speed_arr[0]])
                plt.plot(self.__time_arr, self.__avg_speed_arr,
                         label = "Average speed\n<v> = %.2f m/s" % stats.mean(self.__avg_speed_arr))
                plt.legend()
                plt.show()

            if AVERAGE_VELOCITY_COMPONENTS:
                #Velocity components(t)
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Average velocity component magnitude [m/s]")
                plt.axis([0,self.__time_arr[-1]+self.__time_arr[-1]- self.__time_arr[-2],
                         -stats.mean(self.__avg_speed_arr),stats.mean(self.__avg_speed_arr)])
                plt.plot(self.__time_arr, self.__avg_velocity_x_arr,
                         label = "x-component\n$<v_x>$ = %.2E m/s" % stats.mean(self.__avg_velocity_x_arr))
                plt.plot(self.__time_arr, self.__avg_velocity_y_arr,
                         label = "y-component\n$<v_y>$ = %.2E m/s" % stats.mean(self.__avg_velocity_y_arr))
                plt.legend()
                plt.show()

            if PRESSURE:
                #Pressure(t)
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Pressure [Pa]")
                plt.axis([0,self.__time_arr[-1]+self.__time_arr[-1]- self.__time_arr[-2],
                          0,2*max(self.__pressure_arr)])
                plt.plot(self.__time_arr, self.__pressure_arr,
                         label = "Pressure\n$<P>$ = %.2E Pa" % self.get_average_pressure())
                plt.legend()
                plt.show()

            if TEMPERATURE:
                # Temperature(t)
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Temperature [K]")
                plt.axis([0,self.__time_arr[-1]+self.__time_arr[-1]- self.__time_arr[-2],0,2*self.__temperature_arr[0]])
                plt.plot(self.__time_arr, self.__temperature_arr, label = "Temperature")
                plt.legend()
                plt.show()

            if KE:
                # KE(t)
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Kinetic energy [J]")
                plt.axis([0,self.__time_arr[-1]+self.__time_arr[-1]- self.__time_arr[-2],0,1.5*self.__KE_arr[0]])
                plt.plot(self.__time_arr, self.__KE_arr, label = "Kinetic energy")
                plt.legend()
                plt.show()

            if MOMENTUM:
                # Total momentum of the system components (p_x, p_y)
                plt.rcParams.update(params)
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Momentum components [kgms$^{-1}$]")
                plt.axis([0,self.__time_arr[-1]+self.__time_arr[-1]-self.__time_arr[-2],
                 -abs(3*min([min(self.__momentum_total_x_arr),
                 min(self.__momentum_total_y_arr)])),3*abs(max([max(self.__momentum_total_x_arr), max(self.__momentum_total_y_arr)]))])
                plt.plot(self.__time_arr, self.__momentum_total_x_arr, label = "p$_x$-total")
                plt.plot(self.__time_arr, self.__momentum_total_y_arr, label = "p$_x$-total")
                plt.legend()
                plt.show()
        else:
            pass

    """
    RUNNING THE SIMULATION

    Method run() runs the animation (GUI) by drawing an artist from mathplotlib.pylab.
    By utilizing next_collision() method each frame for the total of num_frames frames,
    the balls collide as they would in a container.
    At the end of the simulation, it displays plots of the time dependent functions and histograms
    by calling get_plots() method.
    """

    def run(self, num_frames, animate, xlim=None, ylim=None, dt_frame=None):

        self.__num_frames = num_frames 
        if animate:
            # To input self.attribute as an optional function argument I have to do the following:
            if (xlim and ylim) is None:
                xlim = (-abs(self._container.radius()), abs(self._container.radius()))
                ylim = (-abs(self._container.radius()), abs(self._container.radius()))
            if dt_frame is None:
                dt_frame = self.__dt_frame
            f = pl.figure()
            ax = pl.axes(xlim=xlim, ylim=ylim, aspect="1")
            ax.add_artist(self.__container_patch)
            for ball in self._balls:
                ax.add_patch(ball.patch())

        for frame in range(num_frames):
            if PROGRESS:
                print(f"Frame: {frame}/{num_frames}. \t Total time passed in the simulation: %.4f s"  % self.__time)
            self.next_collision()
            self.__frame +=1
            if animate:
                pl.pause(dt_frame)
        
        if animate:
            pl.show()

        if PHYSICS:
            self.get_plots()