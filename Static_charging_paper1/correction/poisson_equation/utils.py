import numpy as np
import pandas as pd

def save(filename, a):
    np.savetxt(filename, a.flatten())

def load(filename):
    state = read_dict(filename[:-4]+'_state.txt')
    sim=state['sim']

    shape=(sim['N'],sim['N'],sim['N'])
    shape2=(3,sim['N'],sim['N'],sim['N'])

    phi=np.loadtxt(filename[:-4] + '_potential.txt').reshape(shape)
    E=np.loadtxt(filename[:-4] + '_field.txt').reshape(shape2)
    return E,phi,state

def write_dict(params, filename):
    with open(filename, 'w') as f:
        print(params, file=f)

def read_dict(filename):
    with open(filename, 'r') as f:
        content = f.read()
        return eval(content)