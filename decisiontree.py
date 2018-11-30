# !/usr/bin/env python

import numpy as np
import csv
import sys, os, pdb
from process_data import *
import datetime
class Node(object):
	def __init__(self, feature, T, cp, cm):
	    self.left_child = None
	    self.right_child = None
	    self.threshold = T
	    self.feature = feature
	    if cp>cm:
	    	self.label = 1
	    else:
	    	self.label = -1

    
class DecisionTree():
	def __init__(self, fn_train, fn_valid, fn_test=None):
		self.csv_train = CSV(fn_train)
		self.csv_train.extract_XY()
		self.csv_train.categorize_Y()
		self.x_train, self.y_train = self.csv_train.convert_all_to_numbers()		

		self.csv_valid = CSV(fn_valid)
		self.csv_valid.extract_XY()
		self.csv_valid.categorize_Y()
		self.x_valid, self.y_valid = self.csv_valid.convert_all_to_numbers()		

		if fn_test != None:
			self.csv_test = CSV(fn_test)
			self.csv_test.extract_XY_test()
			self.csv_test.categorize_Y()
			self.x_test, self.y_test = self.csv_test.convert_all_to_numbers()
      
  #Calculate gini index and the benefit of splite feature    
	def gini_benefit(self, cp, cm, clp, clm, crp, crm):
		if clp+clm==0 or crp+crm==0:
			return 0
		else:
			# print "in here"
			pUl = (2*clp*clm/((clp+clm)*(cp+cm)))
			pUr = (2*crp*crm/((crp+crm)*(cp+cm)))
			Ua =  (2*cp*cm/(cp+cm)**2)
			return (Ua-pUl-pUr)

	#split to different leaf
	def split_leaf(self, root_node, root_feature, threshold, para, leaf_clp, leaf_clm, leaf_crp, leaf_crm):
		leaf_clp, leaf_clm, leaf_crp, leaf_crm = 0,0,0,0
		for i in range(0,len(root_node[:])-1):
			y_value = root_node[i] 
			if root_feature[i][para] <= threshold:
				if y_value == 1:
					leaf_clp += 1
				else:
					leaf_clm += 1
			else:
				if y_value == 1:
					leaf_crp += 1
				else:
					leaf_crm += 1
		return leaf_clp, leaf_clm, leaf_crp, leaf_crm

	def count_y(self, y_data):
		cp, cm =0,0
		unique, count = np.unique(y_data, return_counts=True)
		if len(unique) == 2:
			return count[1], count[0]
		elif len(unique) == 0:
			return 0, 0
		elif unique[0] == -1:
			return 0, count[0]
		elif unique[0] == 1:
			return count[0], 0

	def partition(self, y_data, x_data, threshold, feature):
		feature_arr = x_data[:, feature]
		left_leaf_feature = x_data[feature_arr <= threshold]
		left_leaf_value = y_data[feature_arr <= threshold]
		right_leaf_feature = x_data[feature_arr > threshold]
		right_leaf_value = y_data[feature_arr > threshold]
		# pdb.set_trace()
		return left_leaf_feature, left_leaf_value, right_leaf_feature, right_leaf_value

	#Make a single node with left and right
	def make_node(self, root_node, root_feature):
	#initialize the parameters
	#gini
		gini = 0
		gini_f = 0
		gini_temp = 0
		#Threshold    
		T = 0
		T_f = 0
		T_temp = 0

		tree_clp, tree_clm, tree_crp, tree_crm = 0, 0, 0, 0
		best_feature_index = 0

		cp, cm = self.count_y(root_node)
		#Calculate gini-index and benefit
		for j in range(len(root_feature[0])):
			feature_temp = root_feature[:,j]
			for k in range(len(root_node)):
				T_temp = feature_temp[k]
				left = root_node[feature_temp <= T_temp]
				right = root_node[feature_temp > T_temp]
				tree_clp, tree_clm = self.count_y(left)
				tree_crp, tree_crm = self.count_y(right)
				gini_temp = self.gini_benefit(float(cp), float(cm), float(tree_clp), float(tree_clm), float(tree_crp), float(tree_crm))
				if gini_temp > gini_f:
					gini_f= gini_temp
					T_f = T_temp
			if gini_f > gini:
				gini = gini_f
				T = T_f
				best_feature_index = j
		return T, best_feature_index, cp, cm

  #Make a decision tree
	def build_tree(self):
		level = 0
		time_now = datetime.datetime.now()
		# print "lol"
		self.root = self.find_tree(self.y_train,self.x_train,level)
		print "Time taken to build a tree", datetime.datetime.now() - time_now

		print "Starting to predict train data"
		time_now = datetime.datetime.now()
		for i in range(1, 21):
			self.acc_train = self.accuracy(self.y_train, self.x_train, self.root, i)
			print "accuracy at depth {} = {}".format(i, self.acc_train)
		print "Time taken to determine accuracy", datetime.datetime.now() - time_now
		
		print "Starting to predict valid data"
		time_now = datetime.datetime.now()
		for j in range(1, 21):
			self.acc_valid = self.accuracy(self.y_valid, self.x_valid, self.root, j) 	
			print "accuracy at depth {} = {}".format(j, self.acc_valid)
		print "Time taken to determine accuracy", datetime.datetime.now() - time_now	

	def find_tree(self, y_data, x_data, level, max_depth=20):
		# left_feature, right_feature, left_value, right_value, T, real_feature = None, None, None, None, 0,0
		start = datetime.datetime.now()
		T, feature, cp, cm = self.make_node(y_data, x_data)
		print "time spent on node", datetime.datetime.now() - start
		# pdb.set_trace()
		node = Node(feature, T, cp ,cm)
		left_x, left_y, right_x, right_y = self.partition(y_data, x_data, T, feature)
		print "current level",level
		if level <= max_depth-1:
			print len(left_y), len(right_y)
			if len(left_y) > 0:
				node.left_child = self.find_tree(left_y, left_x, level+1)
			if len(right_y) > 0:
				print "in here"
				node.right_child = self.find_tree(right_y, right_x, level+1)
		return node

	def predict(self, x_data_row, node, level, max_level):
		# if node == None:
		# 	return None
		if level >= max_level:
			return node.label
		feature_index = node.feature
		threshold = node.threshold
		if x_data_row[feature_index] <= threshold:
			if node.left_child != None:
				return self.predict(x_data_row, node.left_child, level+1, max_level)
			else:
				return node.label
		else:
			if node.right_child != None:
				return self.predict(x_data_row, node.right_child, level+1, max_level)
			else:
				return node.label

	def accuracy(self, y_data, x_data, root, max_level):
		error = 0
		for i in range(len(y_data)):
			label = self.predict(x_data[i], root, 0, max_level)
			if label != y_data[i]:
				error += 1
				flag.append(i)
		return (float(len(y_data)) - float(error)) / float(len(y_data))




if __name__ == '__main__':
	DT = DecisionTree("pa3_train_reduced.csv", "pa3_valid_reduced.csv") 
	DT.build_tree()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
          
        