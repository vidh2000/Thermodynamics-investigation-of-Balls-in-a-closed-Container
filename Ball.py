"""
@author: Vid Homšak
"""

import matplotlib.pylab as pl
import numpy as np

"""
Default: DEBUG       == 0:
    Each loop bunch of statements are printed out, showing the values stored in certain arrays,
    values of some attributes, local variable values.. which were/can be useful,
    for debugging/checking in the case something doesn't work or gives weird, unexpected results.
"""    
DEBUG = 0

class Ball():
    """
    Ball class describes an object Ball, and contains methods for physical collisions.
    Ball class contains methods, which for two Ball objects say if they will collide, how long it
    will take until next collision and collide the two objects; do the physics of the collision.
    Ball class is then implemented in the Simulation class as the container and in Simulation_types.py
    where appropriate set of balls is created.

    Inputs:
        Floats:       mass, radius
        String:       colour ()
        Lists: position, velocity
    """
    ball_count = 0
    def __init__(self, position, velocity, colour, mass, radius):
        
        # Inputs for vectors need to be of type list and 2D.
        if not isinstance(position, list):
            raise TypeError(f"Position input must be of the type: 'list', but is instead a '{type(position)}'")
        if len(position) != 2:
            raise ValueError(f"We are in a 2D world, position input was {len(position)}D.")
        if not isinstance(velocity, list):
            raise TypeError(f"Velocity input must be of the type: 'list', but is instead a '{type(velocity)}'")
        if len(velocity) != 2:
            raise ValueError(f"We are in a 2D world, velocity input was {len(position)}D.")

        self.__pos                = np.array([position[0], position[1]], dtype = np.float64)
        self.__vel                = np.array([velocity[0], velocity[1]], dtype = np.float64)
        self.__speed              = np.linalg.norm(self.__vel)
        self.__mass               = mass
        self.__colour             = colour
        self.__momentum_x         = self.__mass * self.__vel[0]
        self.__momentum_y         = self.__mass * self.__vel[1]
        self._momentum_change     = 0
        self.__radius             = radius    #Radius for the container is negative!
        self.__patch              = pl.Circle(self.__pos, self.__radius, fc=colour)
        self.__collision_count    = 0
        Ball.ball_count           += 1
        
    # Change the representation of the Ball object so all relevant info about the instance is easily acessible
    def __repr__(self):
        return f'Ball: position = {self.__pos}, velocity = {self.__vel}, mass = {self.__mass}, radius = {self.__radius}'
    
    """
    NAME MANGLING
    Functions that allow to getting/setting values of mangled attributes
    """

    def pos(self):
        return            self.__pos
    
    def vel(self):
        return            self.__vel

    def speed(self):
        return            self.__speed

    def mass(self):
        return            self.__mass
    
    def radius(self):
        return            self.__radius
    
    def momentum_x(self):
        return            self.__momentum_x

    def momentum_y(self):
        return            self.__momentum_y

    def patch(self):
        return            self.__patch

    def move(self, dt):
        """
        Moves the Ball instance in the direction of its velocity for dx = v * dt
        """
        self.__pos       += self.__vel * dt
        self.__patch      = pl.Circle(self.__pos, self.__radius, fc='r')
        
    def time_to_collision(self, other):
        """
        For the two Ball objects, it checks if they will collide or not.
        If they will, then the two possible times are outputted
        (i.e when "front" and when "back" edges of the circle will collide).
        The time of collision for the nearest sides ("front edges") is always chosen == the shortest time.
        In the case that no collision is possible i.e balls' trajectories are skew,
        it prints out the statement saying that (if OUTPUT == 1).
        
        if (they collide):
            return shortest_time
        else:
            pass 
        """

        r           = self.pos() - other.pos()
        v           = self.vel() - other.vel()
        R           = self.radius() + other.radius()
        D           = np.matmul(v,r)*np.matmul(v,r) - np.matmul(v,v)*(np.matmul(r,r) - R*R)
        
        # Checks if they will ever collide or not (does determinant D have any zeros)
        if D > 0:
            quad_coeff = np.array([np.matmul(v,v), 2*np.matmul(v,r), np.matmul(r,r) - R*R])
            solutions = np.roots(quad_coeff)
            #print("Solutions: ", solutions)
            if solutions[0] <=0  and solutions[1] <= 0:
                if DEBUG:
                    print("Objects can collide only if t>0")
                pass
            else:
                shortest_time = min([time for time in solutions if time > 0])
                if DEBUG:
                    print("Shortest time:", shortest_time)
                return shortest_time
        else:
            if DEBUG:
                print("Objects on skew trajectories")
            pass
            
    def collide(self, other):
        """ 
        Does the collision i.e changes the velocities of the two colliding objects
        with respect to how they collided.
        Returns the value of the momenutm change vector magnitude for self and other,
        which will be used to calculate pressure in Simulation.py in the case a
        ball=self collides with the container=other or vice versa.  
        """
        self.__collision_count   += 1
        other.__collision_count  += 1
        
        r                       = self.pos() - other.pos()
        v                       = self.vel() - other.vel()

        # Save momentum momentum values before collision
        momentum_x_prev_self    = self.momentum_x()
        momentum_y_prev_self    = self.momentum_y()

        momentum_x_prev_other   = other.momentum_x()
        momentum_y_prev_other   = other.momentum_y()

        # Update velocity after collision
        self.__vel                = self.vel()-2*other.mass()/(self.mass()+other.mass())*np.matmul(v,r)/np.matmul(r,r)*r
        self.__speed              = np.linalg.norm(self.vel())
        other.__vel               = other.vel()-2*self.mass()/(self.mass()+other.mass())*np.matmul(-v,-r)/ np.matmul(r,r)*(-r)
        other.__speed              = np.linalg.norm(other.vel())
        
        # Update momentum after collision
        self.__momentum_x         = self.mass() * self.vel()[0]
        self.__momentum_y         = self.mass() * self.vel()[1]
        other.__momentum_x        = other.mass() * other.vel()[0]
        other.__momentum_y        = other.mass() * other.vel()[1]

        # Momentum change values
        obj1momentum_change_x   = self.momentum_x() - momentum_x_prev_self
        obj1momentum_change_y   = self.momentum_y() - momentum_y_prev_self
        obj2momentum_change_x   = other.momentum_x() - momentum_x_prev_other
        obj2momentum_change_y   = other.momentum_y() - momentum_y_prev_other
        obj1momentum_change     = abs(np.linalg.norm(np.array([obj1momentum_change_x, obj1momentum_change_y])))
        obj2momentum_change     = abs(np.linalg.norm(np.array([obj2momentum_change_x, obj2momentum_change_y]))) 

        return obj1momentum_change, obj2momentum_change