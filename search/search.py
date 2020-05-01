# coding=utf-8
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Actions
from game import Directions
from util import manhattanDistance
import time

INFINITE_VALUE = float("inf")


class dpQueue:
    def __init__(self):
        self.priority = {}
        self.map_Of_Priorities = {}
        self.minimum_count = 0
        self.minimum_value = None
        self.total_number_vertices = 0

    def size(self):
        return self.total_number_vertices

    def min_state(self):
        return self.minimum_value, self.minimum_count

    def getMin(self):
        if self.size() != 0:
            keys = list(self.priority.keys())
            self.minimum_value = min(keys)
            self.minimum_count = keys.count(self.minimum_value)
        else:
            self.minimum_value = None
            self.minimum_count = 0

    def push(self, value, pKey, sKey):
        if value in self.map_Of_Priorities:
            self.deleteVertex(value)
        self.map_Of_Priorities[value] = (pKey, sKey)
        if pKey not in self.priority:
            self.priority[pKey] = []
        self.priority[pKey].append(value)
        self.total_number_vertices += 1
        if self.minimum_value > pKey or self.minimum_value is None:
            self.minimum_count = 1
            self.minimum_value = pKey
        elif self.minimum_value == pKey:
            self.minimum_count += 1

    def peek(self):
        if 0 >= self.size():
            return None
        result = []
        Keys = self.priority[self.minimum_value]
        for key in Keys:
            prim, sec = self.map_Of_Priorities[key]
            result.append((key, prim, sec))
        result.sort(key=lambda x: x[2])
        return result[0]

    def pop(self):
        result = self.peek()
        if result is not None:
            self.deleteVertex(result[0])
        return result

    def deleteVertex(self, vertex):
        if vertex in self.map_Of_Priorities:
            pKey, _ = self.map_Of_Priorities[vertex]
            self.total_number_vertices -= 1
            del self.map_Of_Priorities[vertex]
            self.priority[pKey].remove(vertex)
            if len(self.priority[pKey]) == 0:
                del self.priority[pKey]
            if self.minimum_value == pKey:
                self.minimum_count -= 1
                if self.minimum_count == 0:
                    self.getMin()


# We have Implemented D*Lite Search Referring to the Psuedo-Code provided in Sven Koenig and Maxim Likhachev.  Dˆ* lite.Aaai/iaai, 15, 2002

class dLiteParametersMock:

    def __init__(self, problem, heuristic_func):
        self.start = problem.getStartState()
        self.prev = problem.getStartState()
        self.goal = problem.getGoalState()
        self.finalPath = []
        self.updatedVertices = []
        self.removedVertices = 0
        self.km = 0
        self.width, self.height = problem.getDims()
        self.pathVertexWeight = [[[INFINITE_VALUE, INFINITE_VALUE] for i in range(self.height)] for j in
                                 range(self.width)]
        self.discoveredWalls = problem.get_walls_discovered_so_far()  # Initially This will be the having the details of the boundaries
        self.heuristic_func = heuristic_func
        self.queue = dpQueue()


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    already_visisted = set()
    curr_state = problem.getStartState()
    dfs_stack = util.Stack()
    dfs_stack.push((curr_state, []))
    next_states = []
    while dfs_stack.isEmpty() != True:
        state = dfs_stack.pop()
        curr_state = state[0]
        next_states = state[1]
        if problem.isGoalState(curr_state) == True:
            return next_states
        if already_visisted.__contains__(curr_state) == False:
            already_visisted.add(curr_state)
            # already_visisted.add(curr_state)
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(states[0]) == False:
                    # print states
                    dfs_stack.push((states[0], next_states + [states[1]]))
    return next_states


def breadthFirstSearch(problem):
    already_visisted = set()
    curr_state = problem.getStartState()
    bfs_queue = util.Queue()
    bfs_queue.push((curr_state, []))
    while bfs_queue.isEmpty() != True:
        state = bfs_queue.pop()
        # print(state)
        curr_state = state[0]
        next_states = state[1]
        # print next_states
        if problem.isGoalState(curr_state) == True:
            # print "In Final"
            # print next_states
            return next_states
        # print(curr_state)
        if already_visisted.__contains__(str(curr_state)) == False:
            already_visisted.add(str(curr_state))
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(str(states[0])) == False:
                    bfs_queue.push((states[0], next_states + [states[1]]))


def uniformCostSearch(problem):
    already_visisted = set()
    curr_state = problem.getStartState()
    bfs_pr_queue = util.PriorityQueue()
    bfs_pr_queue.push((curr_state, []), 0)
    while bfs_pr_queue.isEmpty() != True:
        state = bfs_pr_queue.pop()
        # print state
        curr_state = state[0]
        next_states = state[1]
        # value_curr = state[1]
        if problem.isGoalState(curr_state) == True:
            return next_states
        if already_visisted.__contains__(curr_state) == False:
            already_visisted.add(curr_state)
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(states[0]) == False:
                    curr_cost = problem.getCostOfActions(next_states + [states[1]])
                    bfs_pr_queue.push((states[0], next_states + [states[1]]), curr_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    already_visisted = set()
    curr_state = problem.getStartState()
    bfs_pr_queue = util.PriorityQueue()
    bfs_pr_queue.push((curr_state, []), 0)
    while bfs_pr_queue.isEmpty() != True:
        state = bfs_pr_queue.pop()
        # print state
        curr_state = state[0]
        next_states = state[1]
        # value_curr = state[1]
        if problem.isGoalState(curr_state) == True:
            return next_states
        if already_visisted.__contains__(str(curr_state)) == False:
            already_visisted.add(str(curr_state))
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(str(states[0])) == False:
                    curr_cost = problem.getCostOfActions(next_states + [states[1]])
                    heuristic_cost = heuristic(states[0], problem)
                    bfs_pr_queue.push((states[0], next_states + [states[1]]), curr_cost + heuristic_cost)


def getLocCoordinate(x, y, action):
    dX, dY = Actions.directionToVector(action)
    newX = int(x + dX)
    newY = int(y + dY)
    return newX, newY


def getDirection(cell, nextCell):
    x = cell[0]
    y = cell[1]
    nextX = nextCell[0]
    nextY = nextCell[1]
    if nextX - x > 0:
        return Directions.EAST
    elif nextY - y > 0:
        return Directions.NORTH
    elif x - nextX > 0:
        return Directions.WEST
    else:
        return Directions.SOUTH


def cListAList(locList):
    directions = []
    for cell in range(len(locList) - 1):
        directions.append(getDirection(locList[cell], locList[cell + 1]))
    return directions


def naiveAStarSearch(problem, heuristic):
    start_Time = int(round(time.time() * 1000))
    initailState = problem.getStartState()
    x = initailState[0]
    y = initailState[1]
    finalpath = []
    listOfActions = aStarSearch(problem, heuristic)
    while not problem.isGoalState((x, y)):
        finalpath.append((x, y))
        nextX, nextY = getLocCoordinate(x, y, listOfActions.pop(0))
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            adjX, adjY = getLocCoordinate(x, y, direction)
            if not problem.is_wall_discovered(adjX, adjY) and problem.isWall(adjX, adjY):
                problem.set_walls_discovered_so_far(adjX, adjY)
        if problem.is_wall_discovered(nextX, nextY):
            listOfActions = aStarSearch(problem, heuristic)
            nextX, nextY = getLocCoordinate(x, y, listOfActions.pop(0))
        x, y = nextX, nextY
        problem.setStartState(x, y)
    finalpath.append((x, y))
    actions = cListAList(finalpath)
    problem.setStartState(initailState[0], initailState[1])
    print("Total Time")
    print (int(round(time.time() * 1000)) - start_Time)
    return actions


# Retrives All the Neighbours at a cell
def retrieveNeighbours(dtarParams, node):
    neighbours = []
    for n in [(node[0], node[1] + 1), (node[0], node[1] - 1), (node[0] + 1, node[1]), (node[0] - 1, node[1])]:
        if 0 <= n[1] < dtarParams.height and 0 <= n[0] < dtarParams.width:
            neighbours.append(n)
    return neighbours


def traverseAgent(dtarParams):
    if dtarParams.start == dtarParams.goal:
        return dtarParams.start
    g_start, rhs_start = dtarParams.pathVertexWeight[dtarParams.start[0]][dtarParams.start[1]]
    if g_start == INFINITE_VALUE:
        return dtarParams.start
    cell_weight = (None, INFINITE_VALUE)
    for neighbour in retrieveNeighbours(dtarParams, dtarParams.start):
        weight = 1 + dtarParams.pathVertexWeight[neighbour[0]][neighbour[1]][0]
        if weight < cell_weight[1]:
            cell_weight = (neighbour, weight)
    dtarParams.finalPath.append(dtarParams.start)
    dtarParams.start = cell_weight[0]
    return dtarParams.start


def assignVertexWeight(dtarParams, currentcell, weight):
    if weight[0] is not None:
        dtarParams.pathVertexWeight[currentcell[0]][currentcell[1]][0] = weight[0]
    if weight[1] is not None:
        dtarParams.pathVertexWeight[currentcell[0]][currentcell[1]][1] = weight[1]


def updateVertex(dtarParams, currentCell, ExcludedVertex=None):
    if ExcludedVertex is None:
        ExcludedVertex = dtarParams.start
    if ExcludedVertex != currentCell:
        newRHS = INFINITE_VALUE
        if not dtarParams.discoveredWalls[currentCell[0]][currentCell[1]]:
            for neighbour in retrieveNeighbours(dtarParams, currentCell):
                g, Rhs = dtarParams.pathVertexWeight[neighbour[0]][neighbour[1]]
                newRHS = min(g + 1, newRHS)
        assignVertexWeight(dtarParams, currentCell, (None, newRHS))
    dtarParams.queue.deleteVertex(currentCell)
    g, RHS = dtarParams.pathVertexWeight[currentCell[0]][currentCell[1]]
    if RHS != g:
        pKey, sKey = computeKeys(dtarParams, currentCell)
        dtarParams.queue.push(currentCell, pKey, sKey)


def computeKeys(dtarParams, cellVertex):
    # In this we are using two priority queue for the cells and giving higher weightage for km and heuristic value summation
    sKey = min(dtarParams.pathVertexWeight[cellVertex[0]][cellVertex[1]])
    pKey = dtarParams.km + dtarParams.heuristic_func(dtarParams.start, cellVertex) + sKey
    return pKey, sKey


def compKeys(leftState, rightState):
    if len(leftState) == 2:
        p_1, s_1 = leftState
    elif len(leftState) == 3:
        l_1, p_1, s_1 = leftState
    if len(rightState) == 2:
        p_2, s_2 = rightState
    elif len(rightState) == 3:
        l_2, p_2, s_2 = rightState

    if p_1 < p_2:
        return True
    elif p_1 > p_2:
        return False
    else:
        return s_1 < s_2


def computeShortestPath(dtarParams):
    start_Time = int(round(time.time() * 1000))
    g, rhs = dtarParams.pathVertexWeight[dtarParams.start[0]][dtarParams.start[1]]
    print g, rhs
    while g != rhs or (dtarParams.queue.size() > 0 and compKeys(dtarParams.queue.peek(),
                                                                computeKeys(dtarParams, dtarParams.start))):
        previousvertexvalue = dtarParams.queue.peek()
        vertex = dtarParams.queue.pop()[0]
        dtarParams.removedVertices += 1
        g_vertex, rhs_vertex = dtarParams.pathVertexWeight[vertex[0]][vertex[1]]
        if compKeys(previousvertexvalue, computeKeys(dtarParams, vertex)):
            pKey, sKey = computeKeys(dtarParams, vertex)
            dtarParams.queue.push(vertex, pKey, sKey)
        elif g_vertex > rhs_vertex:
            assignVertexWeight(dtarParams, vertex, (rhs_vertex, None))
            for neighbour in retrieveNeighbours(dtarParams, vertex):
                updateVertex(dtarParams, neighbour, dtarParams.goal)
        else:
            assignVertexWeight(dtarParams, vertex, (INFINITE_VALUE, None))
            for neighbour in retrieveNeighbours(dtarParams, vertex):
                updateVertex(dtarParams, neighbour, dtarParams.goal)
            updateVertex(dtarParams, vertex, dtarParams.goal)
        g, rhs = dtarParams.pathVertexWeight[dtarParams.start[0]][dtarParams.start[1]]


def DStarLiteSearch(problem):
    start_Time = (int(round(time.time() * 1000)))
    print start_Time
    x, y = problem.getStartState()[0], problem.getStartState()[1]
    dStarObj = dLiteParametersMock(problem, manhattanDistance)
    assignVertexWeight(dStarObj, dStarObj.goal, (None, 0))
    pKey, sKey = computeKeys(dStarObj, dStarObj.goal)
    # print pKey,sKey
    dStarObj.queue.push(dStarObj.goal, pKey, sKey)
    computeShortestPath(dStarObj)
    while (x, y) != problem.getGoalState():
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if problem.isWall(nextx, nexty):
                dStarObj.updatedVertices.append((nextx, nexty))
                for neighbour in retrieveNeighbours(dStarObj, (nextx, nexty)):
                    dStarObj.updatedVertices.append(neighbour)
                dStarObj.km += dStarObj.heuristic_func(dStarObj.prev, dStarObj.start)
                dStarObj.prev = dStarObj.start
                dStarObj.discoveredWalls[nextx][nexty] = True
                for updatedVertices in dStarObj.updatedVertices:
                    updateVertex(dStarObj, updatedVertices, dStarObj.goal)
                dStarObj.updatedVertices = []
                computeShortestPath(dStarObj)
        x, y = traverseAgent(dStarObj)
    finalPath = list(dStarObj.finalPath)
    finalPath.append(dStarObj.start)
    finalActions = cListAList(finalPath)
    problem._expanded = dStarObj.removedVertices
    print("Total Time")
    print (int(round(time.time() * 1000)) - start_Time)
    # print (int(round(time.time() * 1000)))
    return finalActions


# Abbreviations
nastar = naiveAStarSearch
dstarlite = DStarLiteSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
