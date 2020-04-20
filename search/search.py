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
from game import Actions, Directions
import dStarLite as dsl
import math


def l1_dist(coord1, coord2): # use manhattan distance from util.py instead?
    p1, p2, q1, q2 = coord1 + coord2
    return abs(p1 - q1) + abs(p2 - q2)


def l2_dist(coord1, coord2):
    p1, p2, q1, q2 = coord1 + coord2
    return math.sqrt(((q1 - p1) ** 2) + ((q2 - p2) ** 2))

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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    setOfVisitedNodes = []  # A list for explored nodes
    fringe = util.Stack()     # DFS uses Stack for Fringe
    fringe.push((problem.getStartState(), []))  # Push(Node, pathTillNode)
    while True:
        popElement = fringe.pop()
        # print(popElement)
        node=popElement[0]
        pathTillNode = popElement[1]

        if problem.isGoalState(node) != False:
            break

        else:
            if node not in setOfVisitedNodes:
                setOfVisitedNodes.append(node)  # Add node to Explored List

                listOfSuccessors = problem.getSuccessors(node)
                for successor in listOfSuccessors:
                    # print(successor)
                    fringe.push(
                        (successor[0], pathTillNode+[successor[1]]))    # Push(Node, pathTillNode) for child node

    return pathTillNode


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    setOfVisitedNodes = []  # A list for explored nodes
    fringe = util.Queue()  # BFS uses Queue for Fringe
    fringe.push((problem.getStartState(), []))  # Push(Node, pathTillNode)
    while True:
        popElement = fringe.pop()
        # print(popElement)
        node = popElement[0]
        pathTillNode = popElement[1]

        if problem.isGoalState(node) != False:
            break

        else:
            if node not in setOfVisitedNodes:
                setOfVisitedNodes.append(node)  # Add node to Explored List

                listOfSuccessors = problem.getSuccessors(node)
                for successor in listOfSuccessors:
                    # print(successor)
                    fringe.push(
                        (successor[0], pathTillNode + [successor[1]]))  # Push(Node, pathTillNode) for child node

    return pathTillNode

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    setOfVisitedNodes = []  # A list for explored nodes
    fringe = util.PriorityQueue()  # UFS uses Priority Queue for Fringe
    fringe.push((problem.getStartState(), [], 0), 0)  # Push(Node, pathTillNode, cost)
    while True:
        popElement = fringe.pop()
        # print(popElement)
        node = popElement[0]
        pathTillNode = popElement[1]
        costTillNode = popElement[2]

        if problem.isGoalState(node) != False:
            break

        else:
            if node not in setOfVisitedNodes:
                setOfVisitedNodes.append(node)  # Add node to Explored List

                listOfSuccessors = problem.getSuccessors(node)
                for successor in listOfSuccessors:
                    # print(successor)
                    fringe.push(
                        (successor[0], pathTillNode + [successor[1]], costTillNode+successor[2]),
                        costTillNode+successor[2])  # Push(Node, pathTillNode, cost) for child node

    return pathTillNode

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

'''
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    setOfVisitedNodes = []  # A list for explored nodes
    fringe = util.PriorityQueue()  # Using Priority Queue Data Structure for Fringe
    fringe.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem) + 0)  # Push(Node, pathTillNode, cost)
    while True:
        popElement = fringe.pop()
        # print(popElement)
        node = popElement[0]
        pathTillNode = popElement[1]
        costTillNode = popElement[2]

        if problem.isGoalState(node) != False:
            break

        else:
            if node not in setOfVisitedNodes:
                setOfVisitedNodes.append(node)  # Add node to Explored List

                listOfSuccessors = problem.getSuccessors(node)
                for successor in listOfSuccessors:
                    # print(successor)
                    fringe.push(
                        (successor[0], pathTillNode + [successor[1]], costTillNode + successor[2]),
                        costTillNode + successor[2] + heuristic(successor[0], problem))  # Push(Node, pathTillNode, cost) for child node

    return pathTillNode
'''


def aStarSearch(problem, heuristic=nullHeuristic):
    already_visisted = set()
    curr_state = problem.getStartState()
    bfs_pr_queue = util.PriorityQueue()
    bfs_pr_queue.push((curr_state, []), 0)
    while bfs_pr_queue.isEmpty() != True:
        state = bfs_pr_queue.pop()
        #print state
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
                    heuristic_cost = heuristic(states[0],problem)
                    bfs_pr_queue.push((states[0], next_states + [states[1]]), curr_cost+heuristic_cost)


def getCoordinate(curr_x, curr_y, action):
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(curr_x + dx), int(curr_y + dy)
    return next_x, next_y


def actionListToCoordList(start_x, start_y, action_list):
    coord_list = []
    for action in action_list:
        coord_list.append((start_x, start_y))
        start_x, start_y = getCoordinate(start_x, start_y, action)
    return coord_list


def getDirection(coord, nextCoord):
    curr_x = coord[0]
    next_x = nextCoord[0]
    curr_y = coord[1]
    next_y = nextCoord[1]
    if curr_x - next_x < 0:
        return Directions.EAST
    elif curr_x - next_x > 0:
        return Directions.WEST
    elif curr_y - next_y < 0:
        return Directions.NORTH
    else:
        return Directions.SOUTH


def coordListToActionList(coord_list):
    directions = []
    for index in range(len(coord_list) - 1):
        coord = coord_list[index]
        nextCoord = coord_list[index + 1]
        direction = getDirection(coord, nextCoord)
        directions.append(direction)
    return directions


def naiveReplanningAStarSearch(problem, heuristic):
    """
    Applies AStarSearch in the scenario where the agent only knows the goal
    state and does not know the location of the walls.  When the agent finds out
    a location of a wall from its successor states, it will need to restart the
    AStarSearch.

    We can initally test the implementation of this algorithm with tinyMaze grid
    using this command:
    python pacman.py -l tinyMaze -p SearchAgent -a fn=nrastar,prob=ReplanningSearchProblem,heuristic=manhattanHeuristic

    Then we can verify that the algorithm works with other grids, using the
    layouts from the layouts/ directory.
    """
    startState = problem.getStartState()
    curr_x, curr_y = startState[0], startState[1]
    pathSoFar = []
    action_list = aStarSearch(problem, heuristic)
    while not problem.isGoalState((curr_x, curr_y)):
        pathSoFar.append((curr_x, curr_y))
        next_x, next_y = getCoordinate(curr_x, curr_y, action_list.pop(0))

        # see the adjacent walls
        for adjacent_direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            adj_x, adj_y = getCoordinate(curr_x, curr_y, adjacent_direction)
            if not problem.isNaiveWall(adj_x, adj_y) and \
                    problem.isWall(adj_x, adj_y):
                problem.setNaiveWalls(adj_x, adj_y)

        # replan only if needed (i.e. we're about to bonk against a wall)
        if problem.isNaiveWall(next_x, next_y):
            action_list = aStarSearch(problem, heuristic)
            next_x, next_y = getCoordinate(curr_x, curr_y, action_list.pop(0))

        curr_x, curr_y = next_x, next_y
        problem.setStartState(curr_x, curr_y)

    pathSoFar.append((curr_x, curr_y))
    actions = coordListToActionList(pathSoFar)
    problem.setStartState(startState[0], startState[1])  # reset the start state, for accurate path cost eval
    return actions


def DStarLiteSearch(problem):
    startState = problem.getStartState()
    x, y = startState[0], startState[1]
    dstarlite_obj = dsl.DStarLite(problem,l1_dist)
    print("goal State")
    print(problem.getGoalState())

    while (x, y) != problem.getGoalState():
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if problem.isWall(nextx, nexty):
                dstarlite_obj.make_wall_at((nextx, nexty))
        x, y = dstarlite_obj.take_step()
    path = dstarlite_obj.get_route()
    directions = coordListToActionList(path)
    problem._expanded = dstarlite_obj._pop_count
    return directions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
nrastar = naiveReplanningAStarSearch
dstarlite = DStarLiteSearch
