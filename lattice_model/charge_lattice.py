import numpy as np



"""
1. randomly place particles on a 2d lattice, no multiple occupancy
2. moving particle has id 1, stuck particle has id 2 on the lattice
3. Create arrays of coords for all particles
4. Each timestep iterate over all particles.
    a. if particle is stuck ignore
    b. if particle is moving, change to stick with probability p
    c. if particle still moving attempt to move randomly up, down, left or right
    d. If new position is empty move particle else do not move and change particle on new site to moving
5. Repeat until all particles are stuck or time runs out.

"""

if __name__ =='__main__':

