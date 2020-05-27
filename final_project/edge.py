

import numpy as np

class Edge:
    def __init__(self,src,dest,measurement, information):
        self.src =src
        self.dest = dest
        self.information = information
        measurement = np.array(measurement).T
        measurement = np.reshape(measurement,(3,1))
        self.measurement = measurement