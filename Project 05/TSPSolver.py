#!/usr/bin/python3

from math import inf
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import copy
import array
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        # pick a random start city, and move through the map going to the cheapest unvisited city at each step
        # try all the cities as start cities before returning inf
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        citiesStarted = 0
        start_time = time.time()
        # create a random permutation
        perm = np.random.permutation(ncities)
        # runs while there hasn't been a solution found and the time limit hasn't been exceeded
        while citiesStarted < ncities and not foundTour and time.time() - start_time < time_allowance:
            route = [cities[perm[0]]]
            perm = perm[1:]
            citiesStarted += 1

            # finds the cheapest "child" city and adds it to the route
            while len(route) < ncities:
                current = route[-1]
                cheapestDist = math.inf
                cheapestCity = None
                for city in cities:
                    if city in route:
                        continue
                    else:
                        if current.costTo(city) < cheapestDist:
                            cheapestDist = current.costTo(city)
                            cheapestCity = city
                if cheapestCity is not None:
                    route.append(cheapestCity)
                else:
                    break

            # checks to see if a complete tour was found
            if len(route) == ncities:
                bssf = TSPSolution(route)
                count += 1
                if bssf.cost < np.inf:
                    # Found a valid route
                    foundTour = True

        # store the results in a dictionary and return the dictionary
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        count = 0
        bssf = None
        max1 = 0
        total = 0
        pruned = 0
        queue = []

        start_time = time.time()

        # get the initial bssf - try the greedy algorithm first, if it doesn't work try the random tour
        # if that doesn't work simly send the cities into the TSPSolver and call it good
        initSoln = self.greedy()
        bssf = initSoln['soln']
        count += initSoln['count']
        if bssf.cost == math.inf:
            initSoln = self.defaultRandomTour()
            bssf = initSoln['soln']
            count += initSoln['count']
            if bssf.cost == math.inf:
                bssf = TSPSolution(cities)

        # create the initial state - state needs to hold only four things: its cost, its matrix, its route and its depth
        initState = self.createInitialState(cities)
        queue.append(initState)
        if len(queue) > max1:
            max1 = len(queue)

        # send the information into the actual branch and bound section of the code
        bssf, max1, total, pruned, count = self.branchAndBoundHelper(cities, bssf, queue, max1, total, pruned, count, start_time, time_allowance)

        # store and return the results
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max1
        results['total'] = total
        results['pruned'] = pruned
        return results

    def branchAndBoundHelper(self, cities, bssf, queue, max1, total, pruned, count, start_time, time_allowance):
        # strore the queue
        S = queue
        # loop through the queue while it still has states and you haven't run out of time
        while len(S) != 0 and time.time() - start_time < time_allowance:
            # pop the highest priority item off the queue
            P = heapq.heappop(S)
            # if it's lowerbound is less that he cost of the BSSF, keep going
            if P.getCost() < bssf.cost:
                # find the cities not in the current route
                unvisited = set(cities) - set(P.getRoute())
                # create a state for each unvisited city and store in a list
                T = [self.createState(P, city, cities) for city in unvisited]
                total += len(T)
                # loop through each of the new states
                for Pi in T:
                    # if we've hit the bottom of the tree, check to see if the route is a cycle and
                    # if it has a lower cost than BSSF. If it does, reassign BSSF based on the new route
                    if len(Pi.getRoute()) == len(cities):
                        count += 1
                        if TSPSolution(Pi.getRoute()).cost < bssf.cost:
                            bssf = TSPSolution(Pi.getRoute())
                    # if we haven't hit the bottom of the tree and the lowerbound is less than BSSF, add to queue
                    elif Pi.getCost() < bssf.cost:
                        heapq.heappush(queue, Pi)
                        if len(queue) > max1:
                            max1 = len(queue)
                    # else, prune the node
                    else:
                        pruned += 1
            # else, prune the node
            else:
                pruned += 1

        # return relevant information
        return bssf, max1, total, pruned, count

    def createInitialState(self, cities):
        # creates the initial state, uses the first city in the list as the starting point in the route

        # first creates a matrix of distances
        matrix = [cities]
        for city in cities:
            row = []
            for city2 in cities:
                row.append(city.costTo(city2))
            matrix.append(row)

        # reduces the matrix and returns the new State
        bound, matrix = self.reduceMatrix(0, matrix)
        return State(bound, matrix, [cities[0]], len(cities))

    def createState(self, P, city, cities):
        # creates a new state based on the parent state

        # makes copies of information it needs from the parents state (these are things that will change)
        bound = copy.deepcopy(P.getCost())
        matrix = copy.deepcopy(P.getMatrix())
        matrix[0] = P.getMatrix()[0]
        parentCityIndex = matrix[0].index(P.getRoute()[-1]) + 1
        childCityIndex = matrix[0].index(city)

        # update the bound based on travel to new state from old state, replace the distance in the matrix with inf
        bound += matrix[parentCityIndex][childCityIndex]
        matrix[parentCityIndex][childCityIndex] = math.inf
        matrix[childCityIndex + 1][parentCityIndex - 1] = math.inf

        # set the from row and to column to inf
        for i in range(len(matrix[0])):
            matrix[parentCityIndex][i] = math.inf
        for i in range(1, len(matrix[0]) + 1):
            matrix[i][childCityIndex] = math.inf

        # reduce the matrix and return the new State
        bound, matrix = self.reduceMatrix(bound, matrix)
        curRoute = [city for city in P.getRoute()]
        curRoute.append(city)
        return State(bound, matrix, curRoute, len(cities))

    def reduceMatrix(self, initBound, matrix):
        # reduces the matrix by making sure that each row and column has a zero, updates the bound

        # start by reducing the rows
        index = 0
        for row in matrix:
            if index == 0:
                index += 1
                continue
            minVal = min(row)
            if minVal != np.inf and minVal != 0:
                for i in range(len(row)):
                    row[i] = row[i] - minVal if row[i] != math.inf else row[i]
                initBound += minVal

        # then reduce the columns
        for i in range(len(matrix[0])):
            col = []
            index = 0
            for row in matrix:
                if index == 0:
                    index += 1
                    continue
                col.append(row[i])
            minVal = min(col)
            if minVal != math.inf and minVal != 0:
                index = 0
                for row in matrix:
                    if index == 0:
                        index += 1
                        continue
                    row[i] = row[i] - minVal if row[i] != math.inf else row[i]
                initBound += minVal

        # return the updated bound and matrix
        return initBound, matrix

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        start_time = time.time()

        numSolutions = 0
        results = {}

        #Initializing basic data
        initialResult = self.greedy()
        greedyBSSF = initialResult['soln']
        initial_route = greedyBSSF.route

        BSSF = initialResult['cost']
        cities = self._scenario.getCities()

        # while (True):
        cityPairs = []
        for source in cities:
            for dest in cities:
                if dest != source and\
                source.costTo(dest) != float("inf"):
                    cityPairs.append((source, dest))
                    
        cityGraph = self.buildGraph(cityPairs)
                    
        cityMatrix = []
        for i in range(len(cities)):
            cityMatrix.append([])
            for j in range(len(cities)):
                cityMatrix[i].append(cities[i].costTo(cities[j]))

        
        path, cost = self.christofides(cityGraph, cityMatrix)

        results['cost'] = cost
        results['time'] = 0
        results['count'] = 1
        results['soln'] = path
        results['max'] = 0
        results['total'] = 0
        results['pruned'] = 0


        return results


    # creates a graph of cities. graph[source][dest] = (distance from source to dest)
    def buildGraph(self, cityPairs):
        graph = {}

        for source, dest in cityPairs:
            if source._index not in graph:
                graph[source._index] = {}
            graph[source._index][dest._index] = source.costTo(dest)

        return graph

    def christofides(self, cityGraph, cityMatrix):

        # build minimum spanning tree in O(n^3) time and O(E) space. Returns a graph of all the paths.
        MinSpanTree = self.minimumSpanningTree(cityGraph)
        print("Minimum Spanning Tree")
        print(MinSpanTree)

        # find odd vertices in O(n^2) time and O(n) space. Returns an array with all the indices of cities with odd degrees.
        oddVertices = self.findOddVertices(MinSpanTree, len(cityGraph))
        unsatisfiedVertices = self.findUnsatisfiedVertices(MinSpanTree, len(cityGraph))
        print("Degrees of vertices")
        print(unsatisfiedVertices)

        # find minimum perfect matching and add to MST
        #There will always be an even number of cities with odd degree so this function works
        cost = float("inf")
        pathsToAdd = {}
        for i in range(100):
            tempCost, tempPathsToAdd = self.findNeededEdges(MinSpanTree, cityMatrix, unsatisfiedVertices.copy())
            if (tempCost < cost):
                cost = tempCost
                pathsToAdd = tempPathsToAdd
        print("Paths to add")
        print(pathsToAdd)
        
        for x in pathsToAdd:
            for y in pathsToAdd[x]:
                if x not in MinSpanTree:
                    MinSpanTree[x] = {}
                MinSpanTree[x][y] = cityMatrix[x][y]
        cost = np.inf
        path = []
        for i in range(20):
            # find eulerian tour
            eulerianTour = self.findEulerianTour(MinSpanTree, len(cityGraph), 0)

            # find hamiltonian circuit
            curPath = self.findHamiltonianCircuit(eulerianTour,  len(cityMatrix), cityMatrix)

            cityPath = (map(lambda city: self._scenario.getCities()[city], curPath))

            if (len(curPath) == 0):
                continue
            curCost = TSPSolution(cityPath).cost
            if (curCost <= cost):
                cost = curCost
                path = cityPath



        # bssf = TSPSolution(path)

        return path, cost

    def minimumSpanningTree(self, cityGraph): #runs in O(n^3) time and O(E) space
        
        cityCount = len(cityGraph)
        
        #Init visited array, including the first city was visited.
        cityWasVisited = [False] * cityCount
        cityWasVisited[0] = True
        
        #Init minimum spanning tree graph.
        MSTgraph = {}
        
        #Quick counter to make sure the loop does not go forever.
        pathCount = 0
        
        while pathCount < cityCount - 1: # O(n) at most
            
            #Breaks if all the cities have been visited.
            if not False in cityWasVisited:
                break
            
            shortestValidPath = np.inf
            startCity = -1
            endCity = -1
            
            #Finds the shortest path from a city that was already visited in O(n^2) time.
            for i in range(cityCount):
                for j in range(cityCount):
                    if j != i and cityWasVisited[i] and not cityWasVisited[j]:
                        if i in cityGraph and j in cityGraph[i] and cityGraph[i][j] < shortestValidPath:
                            startCity = i
                            endCity = j
                            shortestValidPath = cityGraph[i][j]
            
            #If the path exists, add it to the MST.
            if(startCity != -1 and endCity != -1):
                pathCount += 1
                cityWasVisited[endCity] = True
                
                if startCity not in MSTgraph:
                    MSTgraph[startCity] = {}
                MSTgraph[startCity][endCity] = cityGraph[startCity][endCity]
        
        return MSTgraph

    def findOddVertices(self, MinSpanTree, cityCount): #runs in O(n^2) time and O(n) space
        
        #init array to keep track of degrees
        oddDegreeCityDegree = [False] * cityCount
        
        #O(n^2) for loops to find degrees. For every edge, the attached cities increment their degree
        for i in range(cityCount):
            for j in range(cityCount):
                if i in MinSpanTree and j in MinSpanTree[i]:
                    oddDegreeCityDegree[j] = not oddDegreeCityDegree[j]
                    oddDegreeCityDegree[i] = not oddDegreeCityDegree[i]
        
        oddDegreeCities = []
        #O(n) for loop to create an array with all the odd degree city indices
        for i in range(cityCount):
            if oddDegreeCityDegree[i] :
                oddDegreeCities.append(i)
                
        return oddDegreeCities

    def findUnsatisfiedVertices(self, MinSpanTree, cityCount):
        
        cityDegreeArray = [0] * cityCount
        
        for i in range(cityCount):
            for j in range(cityCount):
                if i in MinSpanTree and j in MinSpanTree[i]:
                    cityDegreeArray[i] -= 1
                    cityDegreeArray[j] += 1

        return cityDegreeArray
                    
    def findNeededEdges(self, MinSpanTree, cityMatrix, unsatisfiedVerticies):

        negativeScoreVertices = []
        positiveScoreVertices = []

        pathsToAdd = {}
        cost = 0

        for i in range(len(unsatisfiedVerticies)):
            if unsatisfiedVerticies[i] < 0:
                negativeScoreVertices.append(i)
            if unsatisfiedVerticies[i] > 0:
                positiveScoreVertices.append(i)
                
        import random
        random.shuffle(negativeScoreVertices)
        random.shuffle(positiveScoreVertices)
                
        i = 0
        while i < len(negativeScoreVertices):
            j = 0
            while j < len(positiveScoreVertices):
                
                start = positiveScoreVertices[j]
                end = negativeScoreVertices[i]
                
                if cityMatrix[start][end] != float("inf"):
                    if start not in pathsToAdd:
                        pathsToAdd[start] = {}
                    pathsToAdd[start][end] = cityMatrix[start][end]
                    cost += cityMatrix[start][end]
                    unsatisfiedVerticies[start] -= 1
                    unsatisfiedVerticies[end] += 1
                    if unsatisfiedVerticies[start] == 0:
                        positiveScoreVertices.remove(start)
                        j -= 1
                    if unsatisfiedVerticies[end] == 0:
                        negativeScoreVertices.remove(end)
                        i -= 1
                        break
                        
                j += 1
            i += 1

        if (len(positiveScoreVertices) != 0 or len(negativeScoreVertices) != 0):
            return np.inf, pathsToAdd

        return cost, pathsToAdd

    def minimumMatching(self, MinSpanTree, cityMatrix, oddVertices):
        
        cityCount = len(cityMatrix)
        
        bestLen = np.inf
        bestPath = []
        
        # for i in range(cityCount):
        #     for j in range(cityCount):
                
        
        pass
        
    def findEulerianTour(self, MinSpanTree, cityCount, startNode):
        edge_count = dict()
        for v in MinSpanTree:
            edge_count[v] = len(MinSpanTree[v])
            
        curr_path = []
        circuit = []
        
        curr_path.append(startNode)
        curr_v = startNode
        
        while len(curr_path):
  
            # If there's remaining edge
            if curr_v in edge_count and edge_count[curr_v]:

                # Push the vertex
                curr_path.append(curr_v)

                rand_end_city = random.choice(list(MinSpanTree[curr_v].keys()))

                print("Edge: ", rand_end_city)
                # Find the next vertex using an edge
                next_v = rand_end_city

                # and remove that edge
                edge_count[curr_v] -= 1
                del MinSpanTree[curr_v][rand_end_city]

                # Move to next vertex
                curr_v = rand_end_city

            # back-track to find remaining circuit
            else:
                circuit.append(curr_v)

                # Back-tracking
                curr_v = curr_path[-1]
                curr_path.pop()

        circuit.reverse()

        return circuit

    def findHamiltonianCircuit(self, eulerianTour, cityCount, cityMatrix):

        hasBeenVisited = [False] * cityCount
        path = []

        i = 0
        while i < len(eulerianTour):
            if (i == len(eulerianTour) - 1):
                pass
            elif (hasBeenVisited[eulerianTour[i]]):

                prevNode = eulerianTour[i - 1]
                nextNode = eulerianTour[i + 1]

                if (cityMatrix[prevNode][nextNode] != np.inf):
                    i += 1
                    hasBeenVisited[nextNode] = True
                    path.append(nextNode)
                else:
                    return []
            else:
                hasBeenVisited[eulerianTour[i]] = True
                path.append(eulerianTour[i])
            i += 1
        return path

