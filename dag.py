import numpy as np
from collections import defaultdict
from math import ceil

class Dfg:

	def __init__(self, operations, control_steps, max_j):
		self.source = 0
		self.sink = operations+1
		self.vertices = operations+2 
		self.number_of_operations = operations
		self.graph = defaultdict(list) 
		self.invertedGraph = defaultdict(list)
		self.control_steps = control_steps
		self.range_opers = defaultdict(list)
		self.operations = 1
		self.max_j = max_j
		self.min_j = defaultdict(int)
		self.edges = list()
		self.topSort = []
		self.predecessors = defaultdict(list)
		self.successors = defaultdict(list)
		self.numpreds = defaultdict(list)
		self.numsucc = defaultdict(list)
		self.type_of_operators = dict()
		self.max_oper_type = defaultdict(int)
		self.asap_schedule = []
		self.alap_schedule = []
		self.type_of_operation = dict()
		
	def addEdge(self, u, v, w, tp):
		self.graph[u].append((v,w,tp))
		self.invertedGraph[v].append((u,w,tp))
		self.edges.append((u,v))

		if tp == '$':
			return
		if tp not in self.type_of_operators:
			self.type_of_operators[tp] = self.operations
			self.operations = self.operations+1
		self.max_oper_type[tp] = self.max_oper_type[tp] + 1

		self.type_of_operation[v] = tp

	def calculateMinJ(self):
		for op_type in self.max_oper_type:
			self.min_j[op_type] = int(ceil(self.max_oper_type[op_type]/float(self.control_steps)))

	def printDAG(self):
		print 'Graph'
		print self.graph
		print 'Type'
		print self.type_of_operators
		print 'HC'
		print self.max_j
		print 'Preds:'
		print self.predecessors
		print 'Successors:'
		print self.successors
		print 'NumPreds'
		print self.numpreds
		print 'NumSuccs'
		print self.numsucc
		print 'MaxOperType'
		print self.max_oper_type
		print 'MinJ'
		print self.min_j
		print 'ASAP --> Ignore first and last ( Same for ALAP )'
		print self.asap_schedule
		print 'ALAP'
		print self.alap_schedule
		print 'Total no of types of opers', len(self.type_of_operators)
		print 'Type of Operations'
		print self.type_of_operation

	def dfs(self, v, subtree, graph, params, type_of_fun):

		fix = tuple(x for x in subtree)
		final = subtree

		if v in graph.keys():
			for node, weight, tp in graph[v]:
				subtree = list(x for x in fix)
				self.dfs(node,subtree, graph, params, type_of_fun)
				if type_of_fun == 0 and node != 0:
					subtree.append(node)
				if type_of_fun == 1 and node != self.sink:
					subtree.append(node)
				for x in subtree:
					final.append(x)	

		params[v] = list(set(final))

 	def findPredecessors(self):
 		subtree = []
		self.dfs(self.sink, subtree, self.invertedGraph, self.predecessors, 0)
		for x in self.predecessors:
			self.numpreds[x] = len(self.predecessors[x])

 	def findSuccessors(self):
 		subtree = []
		self.dfs(self.source, subtree, self.graph, self.successors, 1)
		for x in self.successors:
			self.numsucc[x] = len(self.successors[x])



	def topologicalSortUtil(self, v, visited, stack, graph):
		visited[v] = True
		if v in graph.keys():
			for node, weight, tp in graph[v]:
				if visited[node] == False:
					self.topologicalSortUtil(node, visited, stack, graph)

		stack.append(v)

	def topologicalSort(self, s, graph):
		
		#1
		visited = [False]*self.vertices
		stack = []

		#2
		for i in range(self.vertices):
			if visited[i] == False:
				self.topologicalSortUtil(s, visited, stack, graph)

		return stack

	def longestPath(self, s, stack, graph):

		#3
		dist = [float("-Inf")]*(self.vertices)
		dist[s] = 0

		while stack:
			i = stack.pop()

			for node, weight, tp in graph[i]:
				if dist[node] < dist[i]+weight:
					dist[node] = dist[i]+weight

		return dist

	def scheduleASAP(self):
		stack = self.topologicalSort(self.source, self.graph)	
		return self.longestPath(self.source, stack, self.graph)

	def scheduleALAP(self):
		stack = self.topologicalSort(self.sink, self.invertedGraph)
		return self.longestPath(self.sink, stack, self.invertedGraph)

	def evalOperationRange(self):

		asapList = self.scheduleASAP()
		alapList = self.scheduleALAP()
		alapList = [self.control_steps-k for k in alapList]

		self.asap_schedule = asapList
		self.alap_schedule = alapList

		self.range_opers = zip(asapList, alapList)

		print 'O   Range'
		for i in range(1, self.vertices-1):
			print i, self.range_opers[i]



	def printNeurons(self, neurons):

		l = self.number_of_operations+1 # for 1-indexing Convenience
	
		for L in range(1, l):
			J = self.type_of_operation[L]
			J_val = self.type_of_operators[J]
			asap_time = self.asap_schedule[L]
			alap_time = self.alap_schedule[L]

			for i in range(asap_time, alap_time+1):
				for K in range(self.max_j[J]):

					print "( ",i,", ",J_val,", ",K,", ",L, ") has a val ", neurons[i][J_val][K][L] 

	def printOutput(self, output):

		l = self.number_of_operations+1
		for oper in range(1, l):
			flag = False
			asap_time = self.asap_schedule[oper]
			alap_time = self.alap_schedule[oper]

			for i in range(asap_time, alap_time+1):
				J = self.type_of_operation[oper]
				J_val = self.type_of_operators[J]
				for K in range(self.max_j[J]):
					if output[i][J_val][K][oper] == 1:
						flag = True
						print oper, i, K
						break
				if flag == True:
					break

	def run(self):
		print '*********************  Neural Network Scheduling  ************************'

		i = self.control_steps+1
		j = len(self.type_of_operators)+1
		k = max(self.max_j.values())+1
		l = self.number_of_operations+1 # for 1-indexing Convenience
		print 'Initializing (',i,j,k,l,')'
		neurons = np.random.randint(low=-20, high=3, size=(i,j,k,l))
		
		sample_input = raw_input("Press to continue")

		C1 = 1
		C2 = 1
		C3 = 1
		Umax = 20
		Umin = -10
		ite = 1
		list_of_delta_neurons = []

		while(True):
			print 'Iteration ', ite
			# Evaluate Outputs

			output = np.random.randint(low=0, high=1, size=(self.control_steps+1,j,k,l))
			self.printNeurons(neurons)

			for L in range(1, l):

				asap_time = self.asap_schedule[L]
				alap_time = self.alap_schedule[L]
				J = self.type_of_operation[L]
				J_val = self.type_of_operators[J]
				
				maxSoFar = -50
				address = []

				for T in range(asap_time, alap_time+1):
					for Q in range(self.max_j[J]):
						if neurons[T][J_val][Q][L] > maxSoFar:
							maxSoFar =  neurons[T][J_val][Q][L]
							address = []
							address.append(T)
							address.append(Q)

				output[address[0]][J_val][address[1]][L] = 1
			
			self.printNeurons(output)

			# Update Neurons 
			# Calculate /\ U(i,j,k,l)
			list_of_delta_neurons = []
				
			for L in range(1,l):
	
				asap_time = self.asap_schedule[L]
				alap_time = self.alap_schedule[L]
				J = self.type_of_operation[L]
				J_val = self.type_of_operators[J]

				for i in range(asap_time, alap_time+1):
		
					for K in range(self.max_j[J]):

						print 'Working for operation ', L, ' time: ', i,', FU_type: ',K

						inner_term = 0
						for P in range(len(self.predecessors[L])):

							pred_oper =  self.predecessors[L][P]
							pred_operator = self.type_of_operation[pred_oper]
							type_of_pred_oper = self.type_of_operators[pred_operator]
							ap = self.asap_schedule[pred_oper]
							Ap = min(self.alap_schedule[pred_oper], i-1)

							print 'For Pred', pred_oper, 'Running from ', ap,' to ', Ap

							for Q in range(self.max_j[pred_operator]):
								for T in range(ap, Ap+1):
									inner_term = inner_term + output[T][type_of_pred_oper][Q][pred_oper]


						first_term = C1*(self.numpreds[L]-inner_term)

						inner_term = 0
						
						for S in range(len(self.successors[L])):

							succ_oper = self.successors[L][S]
							
							succ_operator = self.type_of_operation[succ_oper]
							type_of_succ_oper = self.type_of_operators[succ_operator]
							As = self.alap_schedule[succ_oper]
							a_s = max(self.asap_schedule[succ_oper], i+1)

							print 'For Successor', succ_oper, ' Running from ', a_s,' to ', As

							for Q  in range(self.max_j[succ_operator]):
								for T in range(a_s, As+1):
									inner_term = inner_term + output[T][type_of_succ_oper][Q][succ_oper]

						second_term = C2*(self.numsucc[L] - inner_term)

						inner_term = -1
						#ww = raw_input()

						for H in range(1,l):
							inner_term = inner_term + output[i][J_val][K][H]


						third_term = C3*inner_term
						
						delta_neuron = (first_term + second_term + third_term)*-1
						
						old_val = neurons[i][J_val][K][L]
						neurons[i][J_val][K][L] = neurons[i][J_val][K][L]+delta_neuron

						if neurons[i][J_val][K][L] > Umax:
							neurons[i][J_val][K][L] = Umax
							delta_neuron = Umax - old_val
						elif neurons[i][J_val][K][L] < Umin:
							neurons[i][J_val][K][L] = Umin
							delta_neuron = Umin - old_val

						list_of_delta_neurons.append(delta_neuron)


			print list_of_delta_neurons


			count = 0
			if ite>1:
				for dn in list_of_delta_neurons:
					if dn == 0:
						count = count + 1

				if count == len(list_of_delta_neurons):
					break

			ite = ite+1

		#self.printNeurons(output)
		self.printOutput(output)		






control_steps = int(raw_input('Enter Control Steps(HW Constraint) : '))

const_filename = raw_input("Enter Constraint filename: ")
f = open(const_filename, "r")
max_j = dict()
for line in f:
	vals = line.strip().split(',')
	max_j[vals[0]] = int(vals[1])	

opers = int(raw_input("Enter no of nodes: "))
g = Dfg(opers, control_steps, max_j)
filename = raw_input("Enter filename: ")
f = open(filename, "r")
for line in f:
	vals = line.strip().split(',')
	g.addEdge(int(vals[0]), int(vals[1]), int(vals[2]), vals[3])

g.calculateMinJ()
g.findPredecessors()
g.findSuccessors()
g.evalOperationRange()
g.printDAG()
g.run()