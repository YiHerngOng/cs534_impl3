# !/usr/bin/env python

import numpy as np
import csv
import sys, os, pdb
import math
from decimal import Decimal

class CSV():
	def __init__(self, filename):
		self.fn = filename
		self.raw = []
		with open(self.fn,'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in reader:
				self.raw.append(row)

	def convert_all_to_numbers(self):
		for i in xrange(len(self.raw)):
			for j in xrange(len(self.x[0])):
				try:
					# pdb.set_trace()
					self.x[i][j] = Decimal(self.x[i][j])
				except:
					raise TypeError
					# try:
					# 	self.arr[i][j] = float(self.arr[i][j])
					# except:
					# 	pass
		return np.array(self.x), np.array(self.y_cat)

	# Fix it later with assignment 1, try not to use dictionary due to complexity
	def categorize(self):
		keys = self.arr[0]
		self.dic = dict()
		for i in xrange(len(keys)):
			temp = []
			for j in range(1, len(self.arr)):
				# print self.arr[j][i]
				temp.append(self.arr[j][i])
			self.dic[keys[i]] = temp

	def extract_XY(self):
		self.x = []
		self.y = []
		for i in xrange(len(self.raw)):
			# if first column is y
			temp = ['1'] + self.raw[i][1:]
			# pdb.set_trace()
			self.x.append(temp)
			self.y.append(self.raw[i][0])

	def categorize_Y(self):
		self.y_cat = []
		for i in xrange(len(self.y)):
			if self.y[i] == '3':
				self.y_cat.append(1)
			elif self.y[i] == '5':
				self.y_cat.append(-1)
		# pdb.set_trace()

