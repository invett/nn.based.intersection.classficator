import numpy as np
import os


for filename in os.listdir('.'):
    if filename.endswith('.npy'):
        input = np.load(filename)
        np.savetxt(filename+'.txt', input)
        
