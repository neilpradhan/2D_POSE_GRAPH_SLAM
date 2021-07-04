
from typing import List
import math
import numpy as np
import node
import edge
from node import Node
from edge import Edge
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import pylab
import matplotlib.pyplot as plt


class graph_slam:
	""" Initializes the nodes and edges and solves the problem of least squres minimization
	taking into account the sptial constraints and problem created by conflicting contraints"""


	def __init__(self):
		self.nodes = [] ## instance attributes
		self.edges = []
		self.total_nodes = 0
		self.total_edges = 0

	# get nodes and edges from dataset
	# make sure one file has only vertices and the other only edges
	def get_from_dataset(self,vertices, edges):
		"""
		Input:
		    vertices: file created by the script.py which contains only vertices
		    edges: file created by the script.py which contains only edges
		Output:
			Fill the self.nodes and self.edges after extracting information
		"""
		with open(vertices, 'r') as v: 
			Lines = v.readlines()
			self.total_nodes = len(Lines)

			for line in Lines:
				numbers  = line.strip("").split(" ")
				numbers = [float(n) for n in numbers[1:]]
				id = numbers[0]
				x,y,yaw = numbers[1:4]
				pose = np.array([[x,y,yaw]])
				self.nodes.append(Node(id,pose))



		with open(edges, 'r') as e: 
			Lines = e.readlines()
			self.total_edges = len(Lines)

			for line in Lines:
				numbers  = line.strip("").split(" ")
				numbers = [float(n) for n in numbers[1:]]
				src = numbers[0]
				dest = numbers[1]
				measurement  = numbers[2:5]


				infm= np.zeros((3,3),dtype = float)
				infm[0,0] = numbers[5] # xx inverse of cov which is information matrix
				infm[1,0] = numbers[6] # xy
				infm[0,1] = numbers[6] #
				infm[0,2] = numbers[7]
				infm[2,0] = numbers[7]
				infm[1,1] = numbers[8]
				infm[2,1] = numbers[9]
				infm[1,2] = numbers[9]
				infm[2,2] = numbers[10]


				self.edges.append(Edge(src,dest,measurement,infm))



	def solve(self):

		"""
		Minimizing the error amongst the conflicting constraints (edges) on the nodes(poses) of the robot using the algorithm
		mentioned in the research paper for solving least squares minimization problem. More information about the algorithm can be obtained from the report
		that is associated with this repository
		"""

		# Initializing the H and b matrices, as both these matrices are mostly sparce,known from their definition
		self.H = np.zeros(shape=(3*self.total_nodes,3*self.total_nodes))
		self.b = np.zeros(shape =(3*self.total_nodes,1))
		
		for e in self.edges:
			source  = e.src
			destination = e.dest
			
			# x_i is the pose (translation) of the robot at a time instance ti
			x_i = self.nodes[int(source)].pose
			x_j = self.nodes[int(destination)].pose

			## finding Aij and Bij 
			
			# Convert into homogenous coordinates for simplicity in explaining various affine transformations between different poses of the robot
			
			T_i = graph_slam.pose_to_hm(x_i) 
			T_j = graph_slam.pose_to_hm(x_j)
			T_ij = graph_slam.pose_to_hm(e.measurement)

			R_i = T_i[0:2,0:2]
			R_ij = T_ij[0:2,0:2] ## relative pose at time j as observed from the post at time i denoted as Rij

			s = math.sin(x_i[2,0]) ## yaw of source
			c = math.cos(x_i[2,0])

			## dow R_i / Dow theta i
			dR_i  = np.array([[-s,c],[-c,-s]]).T 

			dt_ij = x_j[0:2,0]- x_i[0:2,0] ## difference in translation

			dt_ij= np.reshape(dt_ij,(2,1))
			
			
			A_ij = np.vstack((np.hstack((-R_ij.T @ R_i.T , R_ij.T @ dR_i.T @ dt_ij)),np.array([0,0,-1])))


			k = np.array([0,0]).T
			k = np.reshape(k,(2,1))

			B_ij =np.vstack((np.hstack((R_ij.T @ R_i.T , k)),np.array([0,0,1])))

			Z_ij = graph_slam.pose_to_hm(e.measurement) #measurements

			e_ij  = graph_slam.hm_to_pose(np.linalg.inv(Z_ij) @ np.linalg.inv(T_i) @ T_j )
			
			omega_ij = e.information
			H_ii = A_ij.T @ omega_ij @ A_ij
			H_ij = A_ij.T @ omega_ij @ B_ij
			H_ji = B_ij.T @ omega_ij @ A_ij
			H_jj = B_ij.T @ omega_ij @ B_ij
			b_i  = -A_ij.T @ omega_ij @ e_ij
			b_j  = -B_ij.T @ omega_ij  @ e_ij

			# we find the src index and dest index in the big H matrix 
			source = int(source)
			destination = int(destination)
			src_in_H = [(3*source),(3*(source+1))]
			dest_in_H = [(3*destination),(3*(destination+1))]

			p = src_in_H[0]
			q = src_in_H[1]
			r = dest_in_H[0]
			s = dest_in_H[1]		

			self.H[p:q,p:q] +=  H_ii
			self.H[p:q,r:s] +=  H_ij
			self.H[r:s,p:q]+=  H_ij.T
			self.H[r:s,r:s] +=  H_jj
			self.b[p:q]   += b_i
			self.b[r:s]   += b_j

		self.H[0:3,0:3] += np.eye(3) ## fixing only the first index

		H = self.H.copy() ## make a copy
		
		# H = np.mat(H)
		# plt.plot(H)
		# plt.show()
		
		# we convert it into a data structure that handles operations with sparce matrices in a efficient manner
		H_sparse=ss.csc_matrix(H)

		invhs=ssl.splu(H_sparse)

		print("invhs",invhs)

		dx=invhs.solve(self.b)


		print("dx shape ",np.shape(dx))

		## very very important
		## as the dx  = [x1,y1,yaw1,x2,y2,yaw2 .......].T
		delta_pose = dx.reshape((3,self.total_nodes),order='F')


		for i in range(self.total_nodes):
			# print("self.nodes[i_node].pose",np.shape(self.nodes[i_node].pose))
			# print("dx[i_node:i_node+3,1]",np.shape(dx[i_node:i_node+3,0]))
			k = np.reshape(delta_pose[:,i] , (3,1))
			self.nodes[i].pose += k
			assert(self.nodes[i].pose.shape == (3,1))

	@staticmethod 
	def pose_to_hm(pose: List[int])->'numpy_array 3 x 3':
		""" 
		Convert 2d pose (x,y,yaw) into a 3 x 3 homogenous matrix
		reference: http://planning.cs.uiuc.edu/node108.html		
		""" 
		c = math.cos(pose[2,0])
		s = math.sin(pose[2,0])
		hm  = np.array([[c,-s,pose[0,0]],[s,c,pose[1,0]],[0,0,1]])
		return hm

	@staticmethod   
	def hm_to_pose(hm:'numpy array 3x3')->'numpy array 3x1':
		""" 
		Convert homogenous matrix (3 x 3) into 2d pose (x,y,yaw) i.e. (3x1)
		reference: http://planning.cs.uiuc.edu/node108.html		
		""" 
		output  = np.zeros(shape = (3,1))
		output[0,0] = hm[0,2]
		output[1,0] = hm[1,2]
		output[2,0] = math.atan2(hm[1,0],hm[0,0])
		return output

	def plot(self):
		"""
		Plots x versus y to visualize the map of robot through poses
		
		"""
		X=[each_node.pose[0,0] for each_node in self.nodes] ## all x
		Y=[each_node.pose[1,0] for each_node in self.nodes] ## all y
		pylab.plot(X,Y)

	def optimize(self,n_iter=1,vis=False):
		""" 
		Iteratively minimize the conflicting error between several constraints using the algorithm
		and plot the map to see the change and refinement after successful intended iterations
		"""
		for i in range(n_iter):
			print ('Pose Graph Optimization, Iteration %d.'%(i+1))
			self.solve()
			print ('Iteration %d done.'%(i+1))

			if vis:
				pylab.clf()
				self.plot()
				pylab.title('Iteration %d'%(i+1))
				pylab.show()



def main():
	# extract vertices and edges seperately
	vfile='data/intel_vertex.g2o'
	efile='data/intel_edge.g2o'


	pg=graph_slam()
	pg.get_from_dataset(vfile,efile) ## fill the total nodes and total edges and self.nodes and self.edges
	
	# optimize with intended iterations, here iterations = 5
	pg.optimize(5,True)





if __name__ == "__main__":
	main()



	
