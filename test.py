import numpy as np
import math

x = 915
y = 384


COEFF = {
    "a": 0.0000031904,
    "b": 0.00477667,
    "c": 3.419626,
    "d": 303.38164
}
B15 = 314
SCALA_X = (401-128)/(1104-314)
D15 = 128
E13 = 400
FACTOR = 0.109079
B7 = 715



y_2d = int(pow(y, 3)*COEFF["a"]-pow(y, 2)*COEFF["b"]+y*COEFF["c"]-COEFF["d"])
x_2d = int((x-B15)*SCALA_X+D15 - (y_2d-400)*FACTOR*np.sign(x-B7)*(abs(x-B7)>200))
x_2d

