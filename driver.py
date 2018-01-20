import sys
import xlrd
import numpy as np
from copy import deepcopy
from random import randint, sample, random, seed, uniform

class Garden:

	STONE = -1
	
	def __init__(self, matrix, nrows, ncols):
		# map
		self.matrix = matrix
		self.nrows = nrows
		self.ncols = ncols

		# stones
		self.nstones = len(np.argwhere(self.matrix == Garden.STONE))
		self.max_fitness = self.nrows * self.ncols - self.nstones

		# field where it is possible to enter
		self.entrance_fields = []
		for row in range(self.nrows):
			if self.matrix[row][0] != Garden.STONE:
				self.entrance_fields.append([row, -1])
			if self.matrix[row][self.ncols-1] != Garden.STONE:
				self.entrance_fields.append([row, self.ncols])
		for col in range(self.ncols):
			if self.matrix[0][col] != Garden.STONE:
				self.entrance_fields.append([-1, col])
			if self.matrix[self.nrows-1][col] != Garden.STONE:
				self.entrance_fields.append([self.nrows, col])
		self.nentrances = len(self.entrance_fields)

class Monk:

	UP = 1
	DOWN = 2
	LEFT = 3
	RIGHT = 4

	def __init__(self, decisions, entrances):
		# genes determining turn decisions
		# 0 left
		# 1 right
		self.decisions = decisions
		self.ndecisions = len(self.decisions)
		self.decision_index = 0
		# genes determining entering the garden
		# these are indexes of coordinates array within garden matrix
		self.entrances = entrances

	@property
	def decision(self):
		# use decision genes in loops
		decision = self.decisions[self.decision_index]
		self.decision_index += 1
		if self.decision_index >= self.ndecisions:
			self.decision_index = 0
		return decision

	def rake(self, garden):
		self.matrix = deepcopy(garden.matrix)

		marker = 1
		# all paths with different number
		for entrance in self.entrances:
			currentX, currentY = garden.entrance_fields[entrance]

			# set initial direction
			if currentX == -1:
				self.direction = Monk.DOWN
			elif currentX == garden.nrows:
				self.direction = Monk.UP
			elif currentY == -1:
				self.direction = Monk.RIGHT
			elif currentY == garden.ncols:
				self.direction = Monk.LEFT

			while True:
				# step
				nextX, nextY = self.findNext(currentX, currentY)

				if nextX in [-1, garden.nrows] or nextY in [-1, garden.ncols]:
					# out of garden
					break

				if self.matrix[nextX][nextY] != 0:
					# cannot set foot on

					# gene determines rotation direction
					dec = self.decision
					
					if dec == 0:
						nextX, nextY = self.findNext(currentX, currentY, left=True)
					elif dec == 1:
						nextX, nextY = self.findNext(currentX, currentY, right=True)

					if nextX in [-1, garden.nrows] or nextY in [-1, garden.ncols]:
						# out of garden
						break

					if self.matrix[nextX][nextY] != 0:
						# cannot set foot on
	
						if dec == 0:
							nextX, nextY = self.findNext(currentX, currentY, left=True)
						elif dec == 1:
							nextX, nextY = self.findNext(currentX, currentY, right=True)

						if nextX in [-1, garden.nrows] or nextY in [-1, garden.ncols]:
							# out of garden
							break

						if self.matrix[nextX][nextY] != 0:
							# cannot set foot on
	
							if dec == 0:
								nextX, nextY = self.findNext(currentX, currentY, left=True)
							elif dec == 1:
								nextX, nextY = self.findNext(currentX, currentY, right=True)

							if nextX in [-1, garden.nrows] or nextY in [-1, garden.ncols]:
								# out of garden
								break

							if self.matrix[nextX][nextY] != 0:
								# cannot set foot on

								# no more sides
								# monk has got stucked
								self.fitness = 0
								return False
				
				# one step further
				self.matrix[nextX][nextY] = marker
				currentX, currentY = nextX, nextY

			marker += 1
		# task completed
		# calculate fitness of non-stucked monk
		self.fitness = len(np.argwhere(self.matrix > 0))
		return True

	# find next step field, based on direction
	def findNext(self, currentX, currentY, left=False, right=False):
		if left:
			self.left()
		elif right:
			self.right()
		if self.direction == Monk.UP:
			return currentX-1, currentY
		elif self.direction == Monk.DOWN:
			return currentX+1, currentY
		elif self.direction == Monk.LEFT:
			return currentX, currentY-1
		elif self.direction == Monk.RIGHT:
			return currentX, currentY+1

	# turn monk 90 degrees left
	def left(self):
		if self.direction == Monk.UP:
			self.direction = Monk.LEFT
		elif self.direction == Monk.DOWN:
			self.direction = Monk.RIGHT
		elif self.direction == Monk.LEFT:
			self.direction = Monk.DOWN
		elif self.direction == Monk.RIGHT:
			self.direction = Monk.UP

	# turn monk 90 degrees right
	def right(self):
		if self.direction == Monk.UP:
			self.direction = Monk.RIGHT
		elif self.direction == Monk.DOWN:
			self.direction = Monk.LEFT
		elif self.direction == Monk.LEFT:
			self.direction = Monk.UP
		elif self.direction == Monk.RIGHT:
			self.direction = Monk.DOWN

# load excel file
GARDEN_FILE_NAME = "garden.xlsx"
workbook = xlrd.open_workbook(GARDEN_FILE_NAME)
sheet = workbook.sheet_by_index(0)

# build 2D array
nrows = sheet.nrows
ncols = sheet.ncols
matrix = np.array([[int(sheet.cell_value(row, col)) for col in range(ncols)] for row in range(nrows)])

garden = Garden(matrix, nrows, ncols)

with open('output.txt', 'a') as f:
	f.write('Max possible fitness: {}\n'.format(garden.max_fitness))
	print('Max possible fitness: {}'.format(garden.max_fitness))
	f.write('{}\n\n'.format(garden.matrix))
	print(garden.matrix)
	print('', flush=True)


NDECISIONGENES = garden.nstones
NENTRANCEGENES = garden.nrows + garden.ncols

REPEAT_TIMES = 10
MAX_GENERATIONS = 200

for t in range(REPEAT_TIMES):
	seed(1000 * t)
	# first generation of monks able to finish task
	population = []
	while len(population) < 100:
		decisions = []
		for i in range(NDECISIONGENES):
			decisions.append(randint(0,1))
		entrances = sample(range(garden.nentrances), NENTRANCEGENES)
		monk = Monk(decisions, entrances)
		# who is able to finish task survives only
		if monk.rake(garden):
			population.append(monk)

	# best fitness individual ever
	alpha = None
	# iterate generations
	for _ in range(MAX_GENERATIONS-1):

		alpha = max(population, key=lambda attr: attr.fitness)
		#STATISCTICS ONLY
		'''
		if t==2:
		if t==9:
			print(alpha.fitness)
			median = np.median(np.array(list(monk.fitness for monk in population)))
			print(median)
		else:
			continue
		'''
		# END STATISTICS

		if (alpha.fitness == garden.max_fitness):
			# no better individual than this exists
			break
		children = []
		# keep alpha in next generation
		children.append(alpha)

		while len(children) < 100:

			parent1, parent2 = None, None
			if t < REPEAT_TIMES/2:
				# tournament
				s = sample(population, 4)
				parent1 = max(s, key=lambda attr: attr.fitness)
				s = sample(population, 4)
				parent2 = max(s, key=lambda attr: attr.fitness)
			else:
				# roulette
				total = sum(monk.fitness for monk in population)
				m = uniform(0, total)
				n = uniform(0, total)
				current = 0
				for monk in population:
					current += monk.fitness
					if current > m:
						parent1 = monk
					if current > n:
						parent2 = monk
					if parent1 and parent2:
						break

			# decision genes crossover
			cross_decision = int(NDECISIONGENES * random())
			decisions = parent1.decisions[:cross_decision] + parent2.decisions[cross_decision:]
			# decision genes mutation
			while random() < 0.3:
				decisions[randint(0, NDECISIONGENES-1)] = randint(0,1)

			# entrance genes crossover
			cross_entrance = int(NENTRANCEGENES * random())
			entrances = parent1.entrances[:cross_entrance] + parent2.entrances[cross_entrance:]
			# entrance genes mutation
			while random() < 0.3:
				entrances[randint(0, NENTRANCEGENES-1)] = randint(0,garden.nentrances-1)

			monk = Monk(decisions, entrances)
			# who is able to finish task survives only
			if monk.rake(garden):
				children.append(monk)
		# next generation
		population = children

	with open('output.txt', 'a') as f:
		f.write('Achieved: {}\n'.format(alpha.fitness))
		print('Achieved: {}'.format(alpha.fitness))
		f.write('{}\n\n'.format(alpha.matrix))
		print(alpha.matrix)
		print('', flush=True)
