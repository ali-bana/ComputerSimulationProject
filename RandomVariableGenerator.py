import random
import numpy as np
import math
def exponentialGenerator(lamda) :
    r = random.uniform(0, 1)
    return (-1) * (1/ lamda ) * np.log(r)

def poissonGenerator(alpha):
    n = 0
    p = 1
    e = math.exp((-1) * alpha)
    while True:
        r = random.uniform(0, 1)
        p = p * r
        if p < e:
            break
        n += 1
    return n
