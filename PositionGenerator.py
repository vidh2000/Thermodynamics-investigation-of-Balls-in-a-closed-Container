"""
@author: Vid Homšak
"""
import numpy as np

def generate_positions(container_radius, N, alpha):
	"""
	Generates N^2 positions for the balls inside the container.
	Balls' radius is self adjusted so that all the balls fit into the container.
	Alpha < 1 i.e when =1, the balls will not be able to move as they will be "perfectly tightly packed" in the
	initial positions' square lattice.
	"""
	# alpha < 1 condition check
	if alpha >= 1:
		raise ValueError(f"Input parameter alpha = {alpha}, but has to be < 1.")

	# Initial position lattice side
	a = 2 * container_radius/np.sqrt(2)
	
	#Separation between balls
	d = a/N

	# Ball's radius
	r = alpha * d/2

	#Creating positions
	positions = []
	for i in range(N):
		for j in range(N):
			positions.append([-a/2+d/2 + i*d, -a/2+d/2+ j*d])

	return r, positions