
from collections import defaultdict

class Dfg:

	def __init__(self, operations):
		self.vertices = operations+2 
		self.graph = defaultdict(list) 
		self.invertedGraph = defaultdict(list)

	def addEdge(self, u, v, w):
		self.graph[u].append((v,w))
		self.invertedGraph[v].append((u,w))

	def printDAG(self):
		print self.graph
		print self.invertedGraph

	def topologicalSortUtil(self, v, visited, stack, graph):
		print v
		visited[v] = True
		if v in graph.keys():
			for node, weight in graph[v]:
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

		#3
		dist = [float("-Inf")]*(self.vertices)
		dist[s] = 0

		while stack:
			i = stack.pop()

			for node, weight in graph[i]:
				if dist[node] < dist[i]+weight:
					dist[node] = dist[i]+weight

		return dist

	def scheduleASAP(self):
		dist = self.topologicalSort(0, self.graph)	
		print ''
		print '------------------'
		print '-------ASAP-------'
		print '------------------'
		print ''
		for operation in range(1,self.vertices-1):
			print operation, dist[operation]


	def scheduleALAP(self):
		dist = self.topologicalSort(self.vertices-1, self.invertedGraph)
		print ''
		print '------------------'
		print '-------ALAP-------'
		print '------------------'
		print ''		
		for operation in range(1, self.vertices-1):
			print operation, dist[operation]

g = Dfg(11)

f = open("sampleDFG.txt", "r")
for line in f:
	vals = line.strip().split(',')
	g.addEdge(int(vals[0]), int(vals[1]), int(vals[2]))

g.scheduleASAP()
g.scheduleALAP()
