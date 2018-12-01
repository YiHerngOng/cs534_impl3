# !/usr/bin/env python

import numpy as np
import csv
import sys, os, pdb
from process_data import *
import datetime
import math
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

    
class DecisionTree(object):
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

	def count_y(self, y_data, weights): # for adaboost
		# pdb.set_trace()
		cp, cm =0,0
		a = 1
		b = -1
		cp_weight, cm_weight =0, 0
		if len(y_data) == 0:
			return cp_weight, cm_weight
		if a in y_data:
			cp_weight = sum(weights[y_data == 1])
		if b in y_data:
			cm_weight = sum(weights[y_data == -1])

		return cp_weight, cm_weight
		# unique, count = np.unique(y_data, return_counts=True)
		# if len(unique) == 2:
		# 	return float(count[1])*float(cp_weight), float(count[0])*float(cm_weight)
		# elif len(unique) == 0:
		# 	return 0, 0
		# elif unique[0] == -1:
		# 	return 0, count[0]
		# elif unique[0] == 1:
		# 	return count[0], 0

	def partition(self, y_data, x_data, threshold, feature, D_weights):
		feature_arr = x_data[:, feature]
		left_leaf_feature = x_data[feature_arr <= threshold]
		left_leaf_value = y_data[feature_arr <= threshold]
		right_leaf_feature = x_data[feature_arr > threshold]
		right_leaf_value = y_data[feature_arr > threshold]
		left_D_weights = D_weights[feature_arr <= threshold]
		right_D_weights = D_weights[feature_arr > threshold]		
		# pdb.set_trace()
		return left_leaf_feature, left_leaf_value, right_leaf_feature, right_leaf_value, left_D_weights, right_D_weights

	#Make a single node with left and right
	def make_node(self, root_node, root_feature, D_weights):
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

		cp, cm = self.count_y(root_node, D_weights)
		# pdb.set_trace()
		#Calculate gini-index and benefit
		for j in range(len(root_feature[0])):
			feature_temp = root_feature[:,j]
			for k in range(len(root_node)):
				T_temp = feature_temp[k]
				left = root_node[feature_temp <= T_temp]
				right = root_node[feature_temp > T_temp]
				left_weights = D_weights[feature_temp <= T_temp] # y_weights that go to left
				right_weights = D_weights[feature_temp > T_temp] # y_weights that go to right				
				tree_clp, tree_clm = self.count_y(left, left_weights)
				tree_crp, tree_crm = self.count_y(right, right_weights)
				# pdb.set_trace()
				gini_temp = self.gini_benefit(float(cp), float(cm), float(tree_clp), float(tree_clm), float(tree_crp), float(tree_crm))
				# pdb.set_trace()
				if gini_temp > gini_f:
					gini_f= gini_temp
					T_f = T_temp
			if gini_f > gini:
				gini = gini_f
				T = T_f
				best_feature_index = j
		return T, best_feature_index, cp, cm

  #Make a decision tree
	def build_tree(self, D_weights, max_depth):
		level = 0
		time_now = datetime.datetime.now()
		print "Building Tree..."
		self.root = self.find_tree(self.y_train,self.x_train,level, D_weights, max_depth)
		print "Time taken to build a tree", datetime.datetime.now() - time_now

		self.acc_train, error_array = self.accuracy(self.y_train, self.x_train, self.root, max_depth)
		self.acc_valid, error_array = self.accuracy(self.y_valid, self.x_valid, self.root, max_depth) 	
		return error_array

	def find_tree(self, y_data, x_data, level, D_weights, max_depth):
		# left_feature, right_feature, left_value, right_value, T, real_feature = None, None, None, None, 0,0
		start = datetime.datetime.now()
		T, feature, cp, cm = self.make_node(y_data, x_data, D_weights)
		# print "time spent on node", datetime.datetime.now() - start
		# pdb.set_trace()
		node = Node(feature, T, cp ,cm)
		left_x, left_y, right_x, right_y, left_D_weights, right_D_weights = self.partition(y_data, x_data, T, feature, D_weights)
		# print "current level",level
		if level <= max_depth-1:
			# print len(left_y), len(right_y)
			if len(left_y) > 0:
				node.left_child = self.find_tree(left_y, left_x, level+1, left_D_weights, max_depth)
			if len(right_y) > 0:
				# print "in here"
				node.right_child = self.find_tree(right_y, right_x, level+1, right_D_weights, max_depth)
		return node

	def predict(self, x_data_row, node, level, max_level):
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
		error_array = []
		for i in range(len(y_data)):
			label = self.predict(x_data[i], root, 0, max_level)
			if label != y_data[i]:
				error += 1
				error_array.append(i)
		return (float(len(y_data)) - float(error)) / float(len(y_data)), error_array

class adaboost():
	def __init__(self, fn_train, fn_valid, max_depth, L):
		self.fn_train = fn_train
		self.fn_valid = fn_valid
		self.max_depth = max_depth
		self.L = L
		self.DT = DecisionTree(self.fn_train, self.fn_valid)
		print "Number of loops: ", self.L
		print "Maximum depth of tree: ", self.max_depth

	def error_weighted(self, error_location, weighted_vector):
		flag = []
		for i in range(len(self.DT.y_train)):
			if i in error_location:
				flag.append(1)
			else:
				flag.append(0)
		flag = np.array(flag)
		error = np.dot(weighted_vector,flag)
		return error

	def parameter_cal(self, alpha, error_location):
		para_array = []
		para_wrong = math.exp(alpha)
		para_correct = math.exp(-alpha)
		for i in range(len(self.DT.y_train)):
			if i in error_location:
				para_array.append(para_wrong)
			else:
				para_array.append(para_correct)
		return para_array

	def normalize(self, weighted_vector):
		# sum_weighted = np.sum(weighted_vector)
		# for i in range(len(self.DT.y_train)):
		# 	weighted_vector[i] = (float(weighted_vector[i]) - float(min(weighted_vector)))/(float(max(weighted_vector)) - float(min(weighted_vector)))
		# weighted_vector = weighted_vector / np.linalg.norm(weighted_vector)
		weighted_vector = weighted_vector / float(sum(weighted_vector))
		return weighted_vector

	def boosting(self):
		D = []
		parameters = []

		# Initialize D(i) the vector of weights 
		for j in range(len(self.DT.y_train)):
			D.append(float(1)/float(len(self.DT.y_train)))
			# pdb.set_trace()
		D = np.array(D)
		for k in range(self.L):
			time_now = datetime.datetime.now()
			print "Running adaboost loop {}".format(k+1)
			error_sample=self.DT.build_tree(D, self.max_depth)
			training_error=self.error_weighted(error_sample, D)
			print "training_error = ", training_error
			a = (float(1)/float(2))*math.log((float(1)-float(training_error))/float(training_error)) # alpha value
			parameters = self.parameter_cal(a, error_sample) # calc. exponential
			D = np.array(D)
			parameters = np.array(parameters)
			D = D*parameters
			D = self.normalize(D)
			print "Time taken to run a adaboost loop", datetime.datetime.now() - time_now		
			if k+1 == self.L
				print "Current adaboost accuracy with {} loops = training data : {}, validation data: {}".format(k+1, self.DT.acc_train, self.DT.acc_valid)



if __name__ == '__main__':
	# num_adaboost = np.array([1, 5, 10, 20])
	num_adaboost = sys.argv[1]
	max_depth = sys.argv[2]
	A = adaboost("pa3_train_reduced.csv", "pa3_valid_reduced.csv", int(max_depth), int(num_adaboost))
	A.boosting()