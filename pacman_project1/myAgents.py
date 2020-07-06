# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem, mazeDistance

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    def myHeuristic(self, exploring_position, problem):
        foodList = problem.food.asList()
        distance = 0.0

        if len(foodList) == 0:
            return distance

        for food in foodList:

            tmpDistance = ((exploring_position[0] - food[0]) ** 2 + (exploring_position[1] - food[1]) ** 2) ** 0.5

            if tmpDistance > distance:
                distance = tmpDistance

        return distance





    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"

        if len(self.actions) != 0:
            next_action = self.actions[0]
            self.actions = self.actions[1:]
        else:
            startPosition = state.getPacmanPosition(self.index)
            # food = state.getFood()
            # walls = state.getWalls()
            problem = AnyFoodSearchProblem(state, self.index)
            pacman_states = state.getPacmanStates()
            action_list, goal_coords = search.ucs(problem)

            numOfActions = len(action_list)

            for pacman_state in pacman_states:
                if startPosition == pacman_state.configuration.pos:
                    continue
                else:
                    if pacman_state.configuration.direction == 'Stop' and self.actionsTaken != 0:
                        continue
                    else:
                        mazeDist = mazeDistance(pacman_state.configuration.pos, goal_coords, state)

                        if mazeDist <= numOfActions:
                            action_list = ['Stop'] * numOfActions
                            break
            self.actionsTaken += 1
            self.actions = action_list[1:]
            next_action = action_list[0]

        return next_action


    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        self.actions = []
        self.actionsTaken = 0


"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)

        "*** YOUR CODE HERE ***"

        actionList, goalCoords = search.bfs(problem)
        return actionList

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        "*** YOUR CODE HERE ***"
        x, y = state
        food = self.food
        return food[x][y]

