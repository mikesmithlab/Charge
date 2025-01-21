import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    num_sites = 50
    num_collisions = 100000
    
    collisions = np.ones(num_sites)   
    
     
        
    for i in range(num_collisions):
        init_site_rand=np.random.randint(0, high=num_sites-1)
        
        change_site_rand = np.random.uniform()
       
        
        #distribution = np.cos((np.pi/num_sites)*np.linspace(0,num_sites-1, num_sites) + init_site_rand)**2
        collisions_prob = collisions
        
        total = np.sum(collisions_prob)
        prob_bins = np.cumsum(collisions_prob) / np.cumsum(collisions_prob)[-1]
        index = np.argwhere(prob_bins > change_site_rand)[0]
        collisions[index] = collisions[index] + 1
        
            
    plt.figure()
    plt.plot(collisions/np.sum(collisions))
    plt.show()
    
    