# !/usr/bin/env python

import numpy as np
import csv
import sys, os, pdb
from process_data import *
import datetime
import random

class Node(object):
	def __init__(self, feature, T, cp, cm):
	    self.left_child = None
	    self.right_child = None
	    self.threshold = T
	    self.feature = feature

	    if cp>cm:
	    	self.label = 1
	    	# self.percent = float(cp) / (float(cp) + float(cm))
	    else:
	    	self.label = -1
	    	# self.percent = float(cm) / (float(cp) + float(cm))

class RandomForest():
	def __init__(self, num_tree, m_features, depth):
		self.num_tree = num_tree
		self.m_features = m_features
		self.depth = depth

	def build_forest(self, fn_train, fn_valid):
		self.forest = []
		self.DT = DecisionTree(fn_train, fn_valid)

		print "Building forest now"
		time_now = datetime.datetime.now()
		x_train_feature_arr = []
		x_valid_feature_arr = []
		for i in range(self.num_tree):
			x_train_feature_arr = self.sampling_features(self.DT.x_train)
			# print x_train_feature_arr
			self.DT.build_tree(self.DT.x_train, self.depth, x_train_feature_arr)
			self.forest.append(self.DT.root)
			# x_train_feature_arr.append(x_train_feature)
			# x_valid_feature_arr.append(x_valid_feature)
		print "Time taken to build forest", datetime.datetime.now() - time_now
		# pdb.set_trace()
		print "Now calc. accuracy"
		time_now = datetime.datetime.now()
		train_accuracy = self.forrest_accuracy(self.DT.y_train, self.DT.x_train)
		valid_accuracy = self.forrest_accuracy(self.DT.y_valid, self.DT.x_valid)
		print "Time taken to calc accuracy", datetime.datetime.now() - time_now
		print "train_accuracy of {} trees = {}".format(self.num_tree, train_accuracy)
		print "valid_accuracy of {} trees = {}".format(self.num_tree, valid_accuracy)
		return train_accuracy, valid_accuracy

	def forrest_accuracy(self, y_data, x_data):
		error = 0
		for i in range(len(y_data)):
			labels = []
			# percents = []
			for j in range(self.num_tree):
				# pdb.set_trace()
				# print j, i
				label = self.DT.predict_y(x_data[i], self.forest[j], 0, self.depth)
				labels.append(label)
				# percents.append(percent)
			labels = np.array(labels)
			# pdb.set_trace()
			if self.num_tree == 1:
				vote = labels[0]
			else:
				# pdb.set_trace()
				unique, count = np.unique(labels, return_counts = True)
				if len(unique) ==2:
					if count[0] > count[1]:
						vote = unique[0]
					if count[0] < count[1]:
						vote = unique[1]
					if count[0] == count[1]:
						temp = np.random.randint(2)
						if temp == 0:
							vote = unique[0]
						else:
							vote = unique[1]
				else:
					vote = unique[0]
			if vote != y_data[i]:
				error += 1
		return (float(len(y_data)) - float(error)) / float(len(y_data))
		
	#make 10 features subsample		
	def sampling_features(self, x_train):
		# print len(x_train)
		# pdb.set_trace()
		prev = 0
		arr = []
		for i in range(self.m_features):
			index_feature = random.randint(0,99)
			while prev == index_feature:
				index_feature = random.randint(0,99)
			prev = index_feature
			arr.append(index_feature)
			# print index_feature
			# pdb.set_trace()
			# if i == 0:
			# 	x_feature = np.reshape(x_train[:,index_feature], (len(x_train[:,index_feature]), 1))
			# 	x_feature_train = np.reshape(x_valid[:,index_feature], (len(x_valid[:,index_feature]), 1))
			# else:
			# 	x_feature = np.concatenate((x_feature, np.array([x_train[:,index_feature]]).T), axis=1)
			# 	x_feature_train = np.reshape(x_valid[:,index_feature], (len(x_valid[:,index_feature]), 1))

			# x_feature = np.column_stack(x_train[:,index_feature])
			# x_valid_feature = np.concatenate((x_valid[:][index_feature]).T)
		# return x_feature, x_feature_train	
		return np.array(arr)
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

	def count_y(self, y_data):
		cp, cm =0,0
		unique, count = np.unique(y_data, return_counts=True)
		if len(unique) == 2:
			return count[1], count[0]
		if len(unique) == 0:
			return 0, 0
		if len(unique) == 1:
			if unique[0] == -1:
				return 0, count[0]
			if unique[0] == 1:
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
	def make_node(self, root_node, root_feature, m_features=[]):
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
		# for j in range(len(root_feature[0])):
		if len(m_features) == 0:
			temp = range(len(root_feature[0]))
		else:
			temp = m_features
		for j in temp:			
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
	def build_tree(self, x_train, max_depth, m_features=[]):
		time_now = datetime.datetime.now()
		self.root = self.find_tree(self.y_train,x_train,0,max_depth, m_features)
		print "Time taken to build a tree", datetime.datetime.now() - time_now

	def validation_at_each_depth(self, max_level):
		print "Starting to predict train data"
		time_now = datetime.datetime.now()
		for i in range(num_level):
			acc_train = self.accuracy(self.y_train, self.x_train, self.root, i)
			print "accuracy at depth {} = {}".format(i, acc_train)
		print "Time taken to determine accuracy", datetime.datetime.now() - time_now
		
		print "Starting to predict valid data"
		time_now = datetime.datetime.now()
		for j in range(max_level):
			acc_valid = self.accuracy(self.y_valid, self.x_valid, self.root, j) 	
			print "accuracy at depth {} = {}".format(j, acc_valid)
		print "Time taken to determine accuracy", datetime.datetime.now() - time_now	

	def find_tree(self, y_data, x_data, current_level, max_depth, m_features=[]):
		# start = datetime.datetime.now()
		T, feature, cp, cm = self.make_node(y_data, x_data, m_features)
		# print "time spent on node", datetime.datetime.now() - start
		# pdb.set_trace()
		node = Node(feature, T, cp ,cm)
		left_x, left_y, right_x, right_y = self.partition(y_data, x_data, T, feature)
		# print "current level",current_level
		if current_level <= max_depth-1:
			# print len(left_y), len(right_y)
			if len(left_y) > 0:
				node.left_child = self.find_tree(left_y, left_x, current_level+1, max_depth, m_features)
			if len(right_y) > 0:
				# print "in here"
				node.right_child = self.find_tree(right_y, right_x, current_level+1, max_depth, m_features)
		return node

	def predict_y(self, x_data_row, node, current_level, max_depth):
		# print x_data_row
		if current_level >= max_depth:
			return node.label
		feature_index = node.feature
		threshold = node.threshold
		if x_data_row[feature_index] <= threshold:
			if node.left_child != None:
				return self.predict_y(x_data_row, node.left_child, current_level+1, max_depth)
			else:
				return node.label
		else:
			if node.right_child != None:
				return self.predict_y(x_data_row, node.right_child, current_level+1, max_depth)
			else:
				return node.label

	def accuracy(self, y_data, x_data, root, level):
		error = 0
		for i in range(len(y_data)):
			label = self.predict_y(x_data[i], root, 0, level)
			if label != y_data[i]:
				error += 1
		return (float(len(y_data)) - float(error)) / float(len(y_data))

if __name__ == '__main__':
	# DT = DecisionTree("pa3_train_reduced.csv", "pa3_valid_reduced.csv") 
	num_tree = np.array([3])
	# num_tree = np.array([3])
	num_level = 9
	m_features = 10
	# fn = open("rt_accuracy.csv", "wb")
	for i in num_tree:
		RF = RandomForest(i, m_features, num_level)
		train_acc, valid_acc = RF.build_forest("pa3_train_reduced.csv", "pa3_valid_reduced.csv")
		# fn.write(str(train_acc))
		# fn.write(",")
		# fn.write(str(valid_acc))
		# fn.write("\n")
	# fn.close()