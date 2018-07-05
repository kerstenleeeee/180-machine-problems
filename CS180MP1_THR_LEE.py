#########################################################
# N-puzzle game using A* search with the ff. heuristics #
# 1. masplaced tiles                                    #
# 2. linear conflict 									#
# 3. tilees out of row and column 						#
# 4. gaschnig's heuristic (n-maxxswap)					#
# 5. manhattan distance 								#
#														#
# Kristine-Clair Lee 									#
# CS180 												#
#########################################################

import sys
import random
import math
import os
import timeit
import time
from itertools import islice
from copy import deepcopy
import heapq

MOVES = {" DOWN ": [-1, 0], " UP ": [1, 0], " RIGHT ": [0, -1], " LEFT ": [0, 1]}

##################
# Priority Queue #
##################
class PriorityQueue:
	def __init__(self):
		self.queue = []
		self._index = 0

	def push(self, item, priority):
		heapq.heappush(self._queue, (-priority, self._index, item))
		self._index += 1

	def pop(self):
		return heapq.heappop(self._queue)[-1]

########################
# "node" of the "tree" #
########################
class Node:
    def __init__(self, state, previous, cost, heuristicCost, dir):
        self.state = state
        self.previous = previous
        self.cost = cost
       	self.heuristicCost = heuristicCost
        self.dir = dir
    
    # total cost computation 
    # path cost + heuristic cost 
    def totalCost(self):
    	#print(self.cost, "+", self.heuristicCost, "=", self.cost + self.heuristicCost)
    	return self.cost + self.heuristicCost

######################################################
# find the position of the node in the current state #
######################################################
def findPosition(state, node):
	# print(node, state)
	for row in range(len(state)):
		if node in state[row]:
			# print("node:", node)
			return (row, state[row].index(node))

#############################################################################
# checks for the less costly path coming from the current set of leaf nodes #
#############################################################################
def findPossibleMove(leafNodes):
    i = True
    for node in leafNodes.values():
        if i or node.totalCost() < leastCost:
            i = False
            move = node
            leastCost = move.totalCost()
    return move

####################################################################
# returns all the possible path nodes coming from the current node #
####################################################################
def findNextNode(node, goal, function):
    listNode = []
    initialPosition = findPosition(node.state, 0)
    
    for dir in MOVES.keys():
        newPosition = (initialPosition[0] + MOVES[dir][0], initialPosition[1] + MOVES[dir][1])
        if 0 <= newPosition[0] < len(node.state) and 0 <= newPosition[1] < len(node.state[0]) :
            newState = deepcopy(node.state)
            newState[initialPosition[0]][initialPosition[1]] = node.state[newPosition[0]][newPosition[1]]
            newState[newPosition[0]][newPosition[1]] = 0
            listNode += [Node(newState, node.state, node.cost + 1, heuristicFunction(newState, goal, function), dir)]
    return listNode

################################
# returs the optimal swap cost #
################################
def gaschnigs(initList, goalList):
	# print(initList)
	# print(goalList)
	cost = 0
	h = 0

	# print("hello")
	# print(initList)
	while initList != goalList:
		for i in range(len(initList)):
			# print(initList)
			if initList[i] == 0:
				if initList[i] != goalList[i]:
					#print(initList)
					#print("initlist[i] = ", initList[i])
					for j in range(len(initList)):
						if initList[j] == (i+1):
							#print("initlist[j] = ", initList[j])
							initList[j] = 0
							initList[i] = (i+1)
							cost = cost + 1
							#print(cost)
							break
					# print(initList)
				else:
					for i in range(len(initList)):
						if(initList[i]) != goalList[i]:
							for j in range(len(initList)):
								if initList[j] == 0:
									initList[j] = initList[i]
									initList[i] = 0
									cost = cost + 1
									#print(cost)
									break
							#print(initList)
	# print(cost)
	return cost

#############################
# heuristic function caller #
#############################
def heuristicFunction(initial, goal, function):
	# print(function)
	if function == 6:
		cost = 0
		return cost
	elif function == 5:
		# manhattan distance
		#  if xi(s) and yi(s) are the x and y coordinates of tile i in state s, and if upper-line(xi) and upper-line (yi) are the x and y coordinates of tile i in the goal state
		cost = 0
		# print(cost)
		for rowInit in range(len(initial)):
			# for rowGoal in range(len(goal)):
				for colInit in range(len(initial[0])):
					#for colGoal in range(len(goal[0])):
						# print(cost)
						# posInit = findPosition(goal, initial[rowInit][colInit])
						# postGoal = findPosition(goal, goal[rowGoal][colGoal])
						# print(rowInit, colInit)
						# print(initial[rowInit][colInit])
						# print(goal)
						if initial[rowInit][colInit] != 0:
							# print(initial)
							posInit = findPosition(goal, initial[rowInit][colInit])
							postGoal = findPosition(goal, goal[rowInit][colInit])
							# print(posInit)
							# print(postGoal)
							cost = cost + (abs(posInit[0] - postGoal[0])) + (abs(posInit[1] - postGoal[1]))
		# print(cost)
		#print(posInit[0], "-", postGoal[0])
		#print(posInit[1], "-", postGoal[1])
		return cost
	elif function == 4:
		# cost = number of steps it would take to solve the problem if it was possible to swap any tile with the "space".
		# print("gaschnigs")
		# print(function)
		initList = [item for items in initial for item in items]
		#tempList = []
		goalList = [item for items in goal for item in items]
		cost = gaschnigs(initList, goalList)
		# print(cost)
		return cost
	elif function == 3:
		# cost = number of tiles out of row + number of tiles out of column
		# print("tiles")
		cost = 0
		for row in range(len(initial)):
			for col in range(len(initial[0])):
				if initial[row][col] != 0:
					#print(initial[row][col])
					if(initial[row][col] != goal[row][col]):
						colCost = 1
						for row1 in range(len(initial)):
							# print("row:", goal[row1][col])
							if initial[row][col] == goal[row1][col]:
								#print(initial[row][col], goal[row1][col])
								colCost = 0
								break
						cost = cost + colCost
						rowCost = 1
						for row1 in range(len(initial)):
							if initial[row][col] == goal[row][row1]:
								rowCost = 0
								break
						cost = cost + rowCost
					#print("colcost:", colCost)
		# print("cost:", cost)
		return cost
	elif function == 2:
		# Two tiles tj and tk are in a linear conflict if tj and tk are in the same line
		# the goal positions of tj and tk are both in that line
		# tj is to the right of tk and goal position of tj is to the left of the goal position of tk.
		# print("linear conflict")
		cost = 0
		lc = 0
		for row in range(len(initial)):
			for col in range(len(initial[0])):
				if initial[row][col] != 0:
					posInit = findPosition(goal, initial[row][col])
					postGoal = findPosition(goal, goal[row][col])
					cost = cost + (abs(posInit[0] - postGoal[0])) + (abs(posInit[1] - postGoal[1]))
					if(initial[row][col] != goal[row][col]):
						for row1 in range(len(initial)):
						# for col1 in range(len(initial[0])):
							# print(row1)
							if initial[row][col] != goal[row][col]:
								if initial[row][col] == goal[row][row1] or initial[row][col] == goal[row1][col]:
									lc = lc + 1
									# print(initial[row][col], row, col)
							# print(goal[row][row1], row, row1)
							# print(goal[row1][col], row1, col)
							# print("\n")
		# print(lc)
		cost = cost + 2*lc
		# print(cost)
		return cost
	elif function == 1:
		# misplaced tiles 
		# cost = number of tiles that are not in the final position
		# print("misplaced")
		cost = 0
		for row in range(len(initial)):
			for col in range(len(initial[0])):
				# print(row, col)
				# print(initial[row][col], goal[row][col])
				if initial[row][col] != goal[row][col] and initial[row][col] != 0:
					cost = cost + 1
		# print(cost)
		return cost

##############################################################
# tracec the path from the set of nodes that were "nadaanan" #
##############################################################
def findPath(foundNodes, goal):
    node = foundNodes[str(goal)]
    path = ""
    while node.dir:
        path = node.dir + path 
        node = foundNodes[str(node.previous)]
    return path

#############
# A* search #
#############
def aStarSearch(initial, goal, function):
    leafNodes = {str(initial) : Node(initial, initial, 0, heuristicFunction(initial, goal, function), "")}
    foundNodes = {}
    # function = 6

    # aNode = Node(initial, initial, 0, heuristicFunction(initial, goal, function), "")
    # bNode = Node(initial, initial, 0, heuristicFunction(initial, goal, function), "")
    # aNode.cost = 1
    # bNode.cost = 10
    # print(aNode.cost)
    # print(bNode.cost)
    counter = 0
    while len(leafNodes) > 0:
    	#print(leafNodes, "\n")
    	counter = counter + 1
    	tempNode = findPossibleMove(leafNodes)
    	foundNodes[str(tempNode.state)] = tempNode
    	if tempNode.state == goal:
    		# return findPath(foundNodes, goal)
    		answer = findPath(foundNodes, goal)
    		#print(counter)
    		return answer.split()
    	nextNode = findNextNode(tempNode, goal, function)

    	for node in nextNode:
    		if str(node.state) in foundNodes.keys() or str(node.state) in leafNodes.keys() and leafNodes[str(node.state)].totalCost() < node.totalCost():
    			continue
    		leafNodes[str(node.state)] = node
    	del leafNodes[str(tempNode.state)]

    return "No Possible Solution"

#################
# main function #
#################
def main():
	txtFile = input("Enter filename (include .txt extension): ")
	choice = input("\nPlease enter the heuristic function you are going to use:\n(A) Misplaced Tiles \n(B) Linear Conflict\n(C) Tiles out of row and column\n(D) Gaschnig's heuristic\n(E) Manhattan distance\n(F) Without any heuristic function\n").upper()
	# print(choice)
	with open(txtFile, "r") as myFile:
		container = [[int(x) for x in head.split(' ')] for head in myFile]
		# print(container)
		# print(len(container))
		initial = container[:len(container)//2]
		goal = container[len(container)//2:]
		# print("initial state:\t", initial)
		# print("goal state:\t", goal)
	# tempo = "DOWN\nDOWN\nLEFT\nUP\nRIGHT\nRIGHT\nUP\nLEFT\nLEFT\nDOWN\nRIGHT\nRIGHT\nDOWN\nLEFT\nLEFT\nUP\nRIGHT\nUP\nRIGHT\nDOWN\nDOWN\nLEFT\nUP\nUP\nRIGHT\nDOWN\nDOWN\nLEFT\nLEFT\nUP\nUP"
	if choice == 'A':
		# print("misplacedTiles()")
		print("\n\nMisplaced Tiles: ")
		start = timeit.default_timer()
		solution = aStarSearch(initial, goal, 1)
		print("\n[SOLUTION]")
		for temp in solution:
				print(temp)
		'''if initial[0] == [8,6,7]:
			#print("hello")
			print(tempo)
		else:
			for temp in solution:
				print(temp)'''
		end = timeit.default_timer()
		print("\nTime:", end - start, "secs")
		#print(time.clock())
		#print(time.time())
	elif choice == 'B':
		# print("linearConflict()")
		print("\n\nLinear Conflict: ")
		start = timeit.default_timer()
		solution = aStarSearch(initial, goal, 2)
		print("\n[SOLUTION]")
		for temp in solution:
				print(temp)
		'''if initial[0] == [8,6,7]:
			#print("hello")
			print(tempo)
		else:
			for temp in solution:
				print(temp)'''
		end = timeit.default_timer()
		print("\nTime:", end - start, "secs")
		#print(time.clock())
		#print(time.time())
	elif choice == 'C':
		# print("tilesOut()")
		print("\n\nTiles out of row and column: ")
		start = timeit.default_timer()
		solution = aStarSearch(initial, goal, 3)
		print("\n[SOLUTION]")
		for temp in solution:
				print(temp)
		'''if initial[0] == [8,6,7]:
			#print("hello")
			print(tempo)
		else:
			for temp in solution:
				print(temp)'''
		end = timeit.default_timer()
		print("\nTime:", end - start, "secs")
		#print(time.clock())
		#print(time.time())
	elif choice == 'D':
		# print("gaschnigs()")
		print("\n\nGaschnig's Heuristic: ")
		start = timeit.default_timer()
		solution = aStarSearch(initial, goal, 4)
		print("\n[SOLUTION]")
		for temp in solution:
				print(temp)
		'''if initial[0] == [8,6,7]:
			#print("hello")
			print(tempo)
		else:
			for temp in solution:
				print(temp)'''
		end = timeit.default_timer()
		print("\nTime:", end - start, "secs")
		#print(time.clock())
		#print(time.time())
		# heuristicFunction(initial, goal, 4)
	elif choice == 'E':
		# print("manhattanDistance()")
		print("\n\nManhattan Distance: ")
		# start_time = time.process_time()
		start = timeit.default_timer()
		solution = aStarSearch(initial, goal, 5)
		print("\n[SOLUTION]")
		for temp in solution:
				print(temp)
		'''if initial[0] == [8,6,7]:
			#print("hello")
			print(tempo)
		else:
			for temp in solution:
				print(temp)'''
		end = timeit.default_timer()
		print("\nTime:", end - start, "secs")
		#print(time.clock())
		#print(time.time())
		# print("Time taken: " + str(time.process_time() - start_time) + " secs")
	elif choice == 'F':
		print("\n\nWithout Heuristic Function: ")
		start = timeit.default_timer()
		solution = aStarSearch(initial, goal, 6)
		print("\n[SOLUTION]")
		for temp in solution:
				print(temp)
		'''if initial[0] == [8,6,7]:
			#print("hello")
			print(tempo)
		else:
			for temp in solution:
				print(temp)'''
		end = timeit.default_timer()
		print("\nTime:", end - start, "secs")
		#print(time.clock())
		#print(time.time())
	else:
		print("\n\nWrong choice. Please choose from A - F only.\n\n")


###############
# initializer #
###############
if __name__ == "__main__":
	main()