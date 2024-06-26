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

    def __init__(self):
        # Save the history of visited positions for penalizing revisits
        self.positionHistory = set()

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def updatePositionHistory(self, newPos):
        if newPos in self.positionHistory:
            return -100  # Penalty for revisiting a position
        else:
            self.positionHistory.add(newPos)
            return 0  # No penalty

    def evaluationFunction(self, currentGameState, action):
        # Basic information
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()

        # Evaluate distance to nearest food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min(util.manhattanDistance(newPos, food) for food in foodList)
        else:
            minFoodDistance = 0  # No food left

        # Evaluate distance to nearest ghost
        ghostPenalty = 0
        for ghostState in newGhostStates:
            ghostDistance = util.manhattanDistance(newPos, ghostState.getPosition())
            if ghostState.scaredTimer == 0 and ghostDistance < 2:
                ghostPenalty -= 500  # Penalty for getting too close to a ghost
            elif ghostDistance < 2:
                ghostPenalty += 150  # Reward for getting close to a scared ghost

        # Penalty for stopping
        if action == 'Stop':
            stopPenalty = -300
        else:
            stopPenalty = 0

        # Consider the number of legal actions
        numLegalActions = len(childGameState.getLegalActions())
        if numLegalActions > 2:
            mobilityReward = 10
        else:
            mobilityReward = -10  # Penalty for getting stuck

        # Penalty for revisiting a position
        positionPenalty = self.updatePositionHistory(newPos)

        # Bonus for exploration
        explorationBonus = len(self.positionHistory)

        # Evaluate the score
        score = (childGameState.getScore() + 2 * mobilityReward + ghostPenalty
                - minFoodDistance + stopPenalty + positionPenalty + explorationBonus)
        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman's turn (Max player)
                return max_value(agentIndex, depth, gameState)
            else:  # Ghosts' turn (Min player)
                return min_value(agentIndex, depth, gameState)
        
        def max_value(agentIndex, depth, gameState):
            maxValue = float("-inf")
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.getNextState(agentIndex, action)
                value = minimax(1, depth, successor)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
            return bestAction if depth == 0 else maxValue
        
        def min_value(agentIndex, depth, gameState):
            minValue = float("inf")
            nextAgent = agentIndex + 1
            nextDepth = depth + 1 if nextAgent >= gameState.getNumAgents() else depth
            nextAgent %= gameState.getNumAgents()

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.getNextState(agentIndex, action)
                value = minimax(nextAgent, nextDepth, successor)
                if value < minValue:
                    minValue = value
            return minValue
        
        return minimax(0, 0, gameState)

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the action for the alpha-beta agent using self.depth and self.evaluationFunction
        """
        def alpha_beta(agentIndex, depth, gameState, alpha, beta):
            """
            Funcion recursiva de poda alfa-beta para minimax.

            Args:
                agentIndex: indice del agente actual (0 para el jugador, otros indices para adversarios)
                depth: profundidad actual de la recursion
                gameState: estado actual del juego en la recursion
                alpha: valor alfa para la poda
                beta: valor beta para la poda

            Returns:
                El valor de la funcion de evaluacion para los nodos hoja, o el mejor valor calculado para los nodos interiores.
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState, alpha, beta)
            else:
                return min_value(agentIndex, depth, gameState, alpha, beta)
            
        def max_value(agentIndex, depth, gameState, alpha, beta):
            """
            Calcula el valor maximo para los movimientos del jugador.

            Args:
                agentIndex: indice del jugador actual (siempre 0 para el jugador)
                depth: profundidad actual de la recursion
                gameState: estado actual del juego en la recursion
                alpha: valor alfa para la poda
                beta: valor beta para la poda

            Returns:
                El valor maximo encontrado para este nodo o la accion asociada si la profundidad es 0.
            """
            maxValue = float("-inf")
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.getNextState(agentIndex, action)
                value = alpha_beta(1, depth, successor, alpha, beta)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
                alpha = max(alpha, maxValue)
                if maxValue > beta:
                    return maxValue
            return bestAction if depth == 0 else maxValue
        
        def min_value(agentIndex, depth, gameState, alpha, beta):
            """
            Calcula el valor minimo para los movimientos de los fantasmas.

            Args:
                agentIndex: indice del agente adversario
                depth: profundidad actual de la recursion
                gameState: estado actual del juego en la recursion
                alpha: valor alfa para la poda
                beta: valor beta para la poda

            Returns:
                El valor minimo encontrado para este nodo.
            """
            minValue = float("inf")
            nextAgent = agentIndex + 1
            nextDepth = depth + 1 if nextAgent >= gameState.getNumAgents() else depth
            nextAgent %= gameState.getNumAgents()

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.getNextState(agentIndex, action)
                value = alpha_beta(nextAgent, nextDepth, successor, alpha, beta)
                if value < minValue:
                    minValue = value
                beta = min(beta, minValue)
                if minValue < alpha:
                    return minValue
            return minValue
        
        return alpha_beta(0, 0, gameState, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Retorna el mejor movimiento para el agente en el estado actual del juego.

        Todos los fantasmas deben ser modelados como eligiendo uniformemente al azar entre sus movimientos legales.
        
        Args:
            gameState: estado actual del juego
            
        Returns:
            El mejor movimiento para el agente en el estado actual del juego
        """
        actions = gameState.getLegalActions(0)
        maxAction = 'Stop'
        maxResult = float('-inf')
        
        for a in actions:
            # Agente con indice == 0 juega primero
            successor = gameState.getNextState(0, a)
            # Agente con indice == 1 (el primer fantasma) juega siguiente
            currentResult = self.chanceExpect(successor, 0, 1)
            
            if currentResult > maxResult:
                maxResult = currentResult
                maxAction = a
                
        return maxAction
    
    def maxExpect(self, gameState, currDepth):
        """
        Devuelve el valor maximo esperado para un estado de juego y una profundidad dada.

        Args:
            gameState: estado actual del juego
            currDepth: profundidad actual en el arbol de juego

        Returns:
            El valor maximo esperado para el estado del juego
        """
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(0)
        successors = []
        
        for a in actions:
            successors.append( gameState.getNextState(0, a) )
            
        # Agente con indice == 1 (el primer fantasma) juega a continuacion
        valoresEsperados = [self.chanceExpect(s, currDepth, 1) for s in successors]
        
        return max(valoresEsperados)

    def chanceExpect(self, gameState, currDepth, currAgent):
        """
        Devuelve el valor esperado para un estado de juego, una profundidad y un agente dado.
        
        Args:
            gameState: estado actual del juego
            currDepth: profundidad actual en el arbol de juego
            currAgent: agente actual

        Returns:
            El valor esperado para el estado del juego
        """
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(currAgent)
        successors = []
        
        for a in actions:
            # Se obtienen los sucesores del estado actual
            successors.append( gameState.getNextState(currAgent, a) )
            
        if currAgent < gameState.getNumAgents() - 1:
            # Aun hay fantasmas que deben elegir sus movimientos, por lo que se aumenta el indice del agente y se llama a chanceExpect nuevamente
            return sum( [self.chanceExpect(s, currDepth, currAgent + 1 ) for s in successors ] )/len(successors)
        
        else:
            # La profundidad se aumenta cuando es el turno de MAX
            valoresMax = sum([self.maxExpect(s, currDepth + 1 ) for s in successors]) / len(successors)
            return valoresMax


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
