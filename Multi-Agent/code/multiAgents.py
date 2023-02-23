# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        NumberFoodEaten = 100 * (len(currentGameState.getFood().asList()) - len(newFood.asList()))
        ghosts_dist = []
        closestGhost = 0
        for index, ghost_state in enumerate(newGhostStates):
            if newScaredTimes[index] <= 0:
                ghosts_dist.append(manhattanDistance(ghost_state.getPosition(), newPos))
        if ghosts_dist:
            closestGhost = min(ghosts_dist)
        if closestGhost <= 1:
            return -2000000
        closestFood = 0
        foods_dist = []
        for food in newFood.asList():
            foods_dist.append(manhattanDistance(newPos, food))
        if foods_dist:
            closestFood = min(foods_dist)
        return NumberFoodEaten - closestFood + closestGhost


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return solution(self.depth, self.evaluationFunction, 'MinimaxAgent').value_node(gameState, 0)[0]





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
        """

    def getAction(self, gameState):
        return solution(self.depth, self.evaluationFunction, 'AlphaBetaAgent').value_node(gameState, 0)[0]



def getAction(self, gameState):
    """
    Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.value(gameState, 0)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return solution(self.depth, self.evaluationFunction, 'ExpectimaxAgent').value_node(gameState, 0)[0]


class solution:
    def __init__(self, depth, evaluationFunction, type_problem):
        self.depth = depth
        self.evaluationFunction = evaluationFunction
        self.type_problem = type_problem

    def value_node(self, gameState, depth, alpha=-20000, beta=20000):

        End_game_conditions = [gameState.isWin(), depth == self.depth * gameState.getNumAgents(), gameState.isLose()]
        if any(End_game_conditions):
            return ('', self.evaluationFunction(gameState))

        type_agent = depth % gameState.getNumAgents()

        match self.type_problem:
            case 'MinimaxAgent':
                return self.max_function(gameState, depth) if type_agent == 0 \
                    else self.min_function(gameState, depth, type_agent)

            case 'AlphaBetaAgent':
                return self.max_function_alphabeta(gameState, depth, alpha, beta) if type_agent == 0 \
                    else self.min_function_alphabeta(gameState, depth, alpha, beta, type_agent)

            case 'ExpectimaxAgent':
                return self.max_function(gameState, depth) if type_agent == 0 \
                    else self.expectimax_function(gameState, depth, type_agent)

    def min_function(self, gameState, depth, index):
        actions = gameState.getLegalActions(index)
        min_val = ('start', 2000000000000)
        for action in actions:
            succ = gameState.generateSuccessor(index, action)
            res = self.value_node(succ, depth + 1)[1]
            min_val = (action, res) if res < min_val[1] else min_val
        return min_val

    def max_function(self, gameState, depth):
        actions = gameState.getLegalActions(0)
        max_val = ('start', -2000000000000)
        for action in actions:
            succ = gameState.generateSuccessor(0, action)
            res = self.value_node(succ, depth + 1)[1]
            max_val = (action, res) if res > max_val[1] else max_val
        return max_val

    def min_function_alphabeta(self, gameState, depth, alpha, beta, index):
        actions = gameState.getLegalActions(index)
        min_val = ('start', 2000000000000)
        for action in actions:
            succ = gameState.generateSuccessor(index, action)
            res = self.value_node(succ, depth + 1, alpha, beta)[1]
            min_val = (action, res) if res < min_val[1] else min_val
            if min_val[1] < alpha:
                return min_val
            beta = min(beta, min_val[1])
        return min_val

    def max_function_alphabeta(self, gameState, depth, alpha, beta):
        actions = gameState.getLegalActions(0)
        max_val = ('start', -2000000000000)
        for action in actions:
            succ = gameState.generateSuccessor(0, action)
            res = self.value_node(succ, depth + 1, alpha, beta)[1]
            max_val = (action, res) if res > max_val[1] else max_val
            if max_val[1] > beta:
                return max_val
            alpha = max(alpha, max_val[1])
        return max_val

    def expectimax_function(self, gameState, depth, index):
        actions = gameState.getLegalActions(index)
        probability = 1. / len(actions)
        exp_val = 0
        for action in actions:
            succ = gameState.generateSuccessor(index, action)
            res = self.value_node(succ, depth + 1)[1]
            exp_val += res * probability
        return ('', exp_val)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
