#!/usr/bin/python3

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
        pass
