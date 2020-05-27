import math
import numpy as np
from typing import List






class Node:
    def __init__(self, id ,pose):
        self.pose = np.array(pose).T
        self.x = self.pose[0,0]
        self.y = self.pose[1,0]
        self.yaw = self.pose[2,0]










