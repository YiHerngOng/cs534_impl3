#!/usr/bin/env python
import numpy as np
import os, sys, pdb
from process_data import *


class Decision_Tree(object):
	def __init__(self, fn_train, fn_valid):
		self.csv_train = CSV(fn_train)
		self.csv_train.extract_XY()
		self.csv_train.categorize_Y()
		self.x_train, self.y_train = self.csv_train.convert_all_to_numbers()		

		self.csv_valid = CSV(fn_valid)
		self.csv_valid.extract_XY()
		self.csv_valid.categorize_Y()
		self.x_valid, self.y_valid = self.csv_valid.convert_all_to_numbers()


	def gini_index(self):
		

if __name__ == '__main__':
	DT = Decision_Tree("pa3_train_reduced.csv", "pa3_valid_reduced.csv")
	