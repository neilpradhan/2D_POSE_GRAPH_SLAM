
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


	def __init__(self):
		self.nodes = [] ## instance attributes
		self.edges = []
		self.total_nodes = 0
		self.total_edges = 0

	# get nodes and edges from dataset
	# make sure one file has only vertices and the other only edges

	
	def get_from_dataset(self,vertices, edges):
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
			# print(total_nodes)
			# print(self.nodes)


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
			# print(total_edges)
			# print(self.edges)




	def solve(self):

		# assert (self.total_nodes!=[])
		# print(self.total_nodes)

		self.H = np.zeros(shape=(3*self.total_nodes,3*self.total_nodes))
		self.b = np.zeros(shape =(3*self.total_nodes,1))
		
		for e in self.edges:
			source  = e.src
			destination = e.dest
			x_i = self.nodes[int(source)].pose
			x_j = self.nodes[int(destination)].pose

			## finding Aij and Bij 
			T_i = graph_slam.pose_to_hm(x_i) 
			T_j = graph_slam.pose_to_hm(x_j)
			T_ij = graph_slam.pose_to_hm(e.measurement)


			# print("T_ij",T_ij)
			# break

			R_i = T_i[0:2,0:2]
			R_ij = T_ij[0:2,0:2] ## relative pose with j with respect to i Rij

			s = math.sin(x_i[2,0]) ## yaw of source
			c = math.cos(x_i[2,0])

			## dow R_i / Dow theta i
			dR_i  = np.array([[-s,c],[-c,-s]]).T 

			dt_ij = x_j[0:2,0]- x_i[0:2,0] ## difference in translation

			dt_ij= np.reshape(dt_ij,(2,1))


			# print("R_ij size",np.shape(R_ij))
			# print("R_i.T size", np.shape(R_i.T))
			# print("dR_i.T shape",np.shape(dR_i.T))
			# print("-R_ij.T * R_i.T shape", np.shape(-R_ij.T * R_i.T))
			# print("R_ij.T * dR_i.T * dt_ij",np.shape(R_ij.T * dR_i.T * dt_ij))
			# # print("dt_ij", np.shape(dt_ij))			
			# print("dt_ij", np.shape(dt_ij))
			# print("R_ij.T @ R_i.T", np.shape(R_ij.T @ R_i.T))
			# print("np.array([0,0]).T", np.shape(np.array([0,0]).T))			
			# A_ij =np.concatenate((np.concatenate((-R_ij.T * R_i.T , R_ij.T * dR_i.T * dt_ij),axis=1),np.array([0,0,-1])),axis=0)
			
			
			A_ij = np.vstack((np.hstack((-R_ij.T @ R_i.T , R_ij.T @ dR_i.T @ dt_ij)),np.array([0,0,-1])))
			# print("A",A_ij)
			# break

			k = np.array([0,0]).T
			k = np.reshape(k,(2,1))

			B_ij =np.vstack((np.hstack((R_ij.T @ R_i.T , k)),np.array([0,0,1])))

			# print("B",B_ij)
			# break

			Z_ij = graph_slam.pose_to_hm(e.measurement) #measurements

			e_ij  = graph_slam.hm_to_pose(np.linalg.inv(Z_ij) @ np.linalg.inv(T_i) @ T_j )
			
			omega_ij = e.information
			# print("omega_ij",omega_ij)
			# break
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

			# print(p,q,r,s)
			

			# print("src_in_H[0]",src_in_H[0])
			# print("src_in_H[0]",src_in_H[1])
			# print("H_ii", np.shape(H_ii))
			# print("shape 1", np.shape(self.H[src_in_H[0]:src_in_H[1],src_in_H[0]:src_in_H[1]]))
			# print("my ", np.shape(self.H))
			# print(src_in_H[0],src_in_H[1])

			self.H[p:q,p:q] +=  H_ii
			self.H[p:q,r:s] +=  H_ij
			# self.H[r:s,p:q] +=  H_ji
			self.H[r:s,p:q]+=  H_ij.T
			self.H[r:s,r:s] +=  H_jj




			self.b[p:q]   += b_i
			self.b[r:s]   += b_j

			
			
			# print("b_i", b_i)
			# print()

			# print("b_j",b_j)
			# print()

			# print("H_ij", H_ij)
			# print()


			# print("H_ji", H_ji)
			# break

			# print("H_ii",H_ii)
			# print()


			# print("e_ij",e_ij)
			# print()

			# print("Z_ij",Z_ij)
			# print()
			# plt.plot(self.H)
			# plt.show()
			# break

			# plt.plot(self.b)
			# plt.show()
			# break



		self.H[0:3,0:3] += np.eye(3) ## fixing only the first index

		H = self.H.copy() ## make a copy
		
		# H = np.mat(H)
		# plt.plot(H)
		# plt.show()
		
		#make a sparce matrix
		H_sparse=ss.csc_matrix(H)

		# print("Hsparce",H_sparse[2][0])
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
			# print()				
			# print("updated_pose", self.nodes[i_node].pose)
			# break













	## pose to homogeneous cordinates
	@staticmethod 
	def pose_to_hm(pose: List[int])->'numpy_array 3 x 3':
		# hm = np.zeros(shape = (3,3), type =float)

		### planning
		# http://planning.cs.uiuc.edu/node108.html

		c = math.cos(pose[2,0])
		s = math.sin(pose[2,0])
		hm  = np.array([[c,-s,pose[0,0]],[s,c,pose[1,0]],[0,0,1]])
		return hm



	@staticmethod   
	def hm_to_pose(hm:'numpy array 3x3')->'numpy array 3x1':
		output  = np.zeros(shape = (3,1))
		output[0,0] = hm[0,2]
		output[1,0] = hm[1,2]
		output[2,0] = math.atan2(hm[1,0],hm[0,0])
		return output

	def plot_(self):
		plt.figure(1)
		plt.title("M3500_A dataset 100 iterations of optimization", fontsize=10)
		X=[each_node.pose[0,0] for each_node in self.nodes] ## all x
		Y=[each_node.pose[1,0] for each_node in self.nodes] ## all y
		pylab.plot(X,Y)

	def plot(self, i, arr, axes):

		X=[each_node.pose[0,0] for each_node in self.nodes] ## all x
		Y=[each_node.pose[1,0] for each_node in self.nodes] ## all y
		axes[arr[i][0],arr[i][1]].plot(X,Y)


	def optimize(self,n_iter=1,vis=False):

		fig, axes = plt.subplots(nrows = 10,ncols = 10)
		plt.legend()
		fig.suptitle("M3500_A dataset 100 iterations of optimization", fontsize=10)
		arr = [[i,j] for i in range(10) for j in range(10)]
		for i in range(n_iter):
			print ('Pose Graph Optimization, Iteration %d.'%(i+1))
			self.solve()
			print ('Iteration %d done.'%(i+1))
			
			if vis:
				# pylab.clf()
				#pylab.ion()

				self.plot(i,arr, axes)
				# pylab.title('Iteration %d'%(i+1))
				#pylab.draw()
				#pylab.ioff()
				#time.sleep(3)
		# plt.title("9 iterations of optimization"	)
		# plt.subplots_adjust(top=0.85)
		plt.tight_layout(rect = [0, 0.03, 1, 0.95])


		# pylab.show()
		pylab.clf()
		self.plot_()
		pylab.show()


 
def main():
	# print("Hello World!")
	# g = graph_slam()

	# pose = np.array([[1,2,3.14]])

	# print(pose[0,0])
	# a = g.pose_to_hm(pose)
	# print(a)


	# vertices = 'data/intel_vertex.g2o'
	# edges  = 'data/intel_edge.g2o'

	# vertices = 'data/vertices_M3500.g2o'
	# edges  = 'data/edges_M3500.g2o'

	# vertices = 'data/vertices_MITb.g2o'
	# edges = 'data/edges_MITb.g2o'

	vertices = 'data/vertices_M3500a.g2o'
	edges = 'data/edges_M3500a.g2o'

	# vertices = 'data/vertices_M3500b.g2o'
	# edges = 'data/edges_M3500b.g2o'

	# vertices = 'data/vertices_M3500c.g2o'
	# edges = 'data/edges_M3500c.g2o'


	pg=graph_slam()
	pg.get_from_dataset(vertices,edges) ## fill the total nodes and total edges and self.nodes and self.edges
	pg.optimize(100,True)





if __name__ == "__main__":
	main()



	