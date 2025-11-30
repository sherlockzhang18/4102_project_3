# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        
        for _ in range(self.iterations):
            newValues = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0.0
                    continue

                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    newValues[state] = 0.0
                    continue

                bestValue = float('-inf')
                for action in actions:
                    qValue = 0.0
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        qValue += prob * (reward + self.discount * self.values[nextState])
                    if qValue > bestValue:
                        bestValue = qValue

                newValues[state] = bestValue

            self.values = newValues



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        qValue = 0.0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue


    def computeActionFromValues(self, state):

        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None

        bestAction = None
        bestValue = float('-inf')

        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action

        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        if not states:
            return

        numStates = len(states)

        for i in range(self.iterations):
            state = states[i % numStates]

            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            if not actions:
                continue

            bestValue = float('-inf')

            for action in actions:
                qValue = 0.0
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action, nextState)
                    qValue += prob * (reward + self.discount * self.values[nextState])
                if qValue > bestValue:
                    bestValue = qValue

            self.values[state] = bestValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        
        states = self.mdp.getStates()
        predecessors = {}
        for s in states:
            predecessors[s] = set()

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    if prob > 0:
                        predecessors.setdefault(nextState, set()).add(s)

        pq = util.PriorityQueue()

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            if not actions:
                continue

            maxQ = max(self.computeQValueFromValues(s, a) for a in actions)
            diff = abs(self.values[s] - maxQ)

            pq.update(s, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()

            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                if actions:
                    bestValue = max(self.computeQValueFromValues(s, a) for a in actions)
                    self.values[s] = bestValue

            for p in predecessors.get(s, []):
                if self.mdp.isTerminal(p):
                    continue
                actions = self.mdp.getPossibleActions(p)
                if not actions:
                    continue

                maxQ = max(self.computeQValueFromValues(p, a) for a in actions)
                diff = abs(self.values[p] - maxQ)

                if diff > self.theta:
                    pq.update(p, -diff)


