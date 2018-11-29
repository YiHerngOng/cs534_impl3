# !/usr/bin/env python

import numpy as np
import csv
import sys, os, pdb
from process_data import *
import datetime
class Node(object):
	def __init__(self, feature, T, y_values):
	    self.left_child = None
	    self.right_child = None
	    self.threshold = T
	    self.feature = feature
	    unique, count = np.unique(y_values, return_counts=True)
	    count_1 = count[1] # count how many 3
	    count_neg1 = count[0] # count how many 5
	    if count_1 > count_neg1:
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
		# left_leaf_value = []
		# right_leaf_value = []
		# left_leaf_feature = []
		# right_leaf_feature = []
		leaf_clp, leaf_clm, leaf_crp, leaf_crm = 0,0,0,0
		for i in range(0,len(root_node[:])-1):
			y_value = root_node[i] 
			if root_feature[i][para] <= threshold:
				# for j in range(0,len(root_feature[i][:])-1):
				# 	left_leaf_feature.append(root_feature[i][j])
				# left_leaf_value.append(root_node[i])
				if y_value == 1:
					leaf_clp += 1
				else:
					leaf_clm += 1
			else:
				# for k in range(0,len(root_feature[i][:])):
				# 	right_leaf_feature.append(root_feature[i][k])
				# right_leaf_value.append(root_node[i])
				if y_value == 1:
					leaf_crp += 1
				else:
					leaf_crm += 1
		return leaf_clp, leaf_clm, leaf_crp, leaf_crm

	def count_y(self, y_data):
		cp, cm =0,0
		unique, count = np.unique(y_data, return_counts=True)
		if len(unique) == 2:
			cm = count[0]
			cp = count[1]
		elif len(unique) == 0:
			return cp, cm
		elif unique[0] == -1:
			cm = count[0]
		elif unique[0] == 1:
			cp = count[0]
		return cp, cm

	def partition(self, y_data, x_data, threshold, feature):
		for i in range(len(y_data)):
			left_leaf_value = []
			right_leaf_value = []
			left_leaf_feature = []
			right_leaf_feature = []	
			if x_data[i][feature] <= threshold:
				left_leaf_feature.append(x_data[i][feature])
				left_leaf_value.append(y_data[i])
			else:
				right_leaf_feature.append(x_data[i][feature])
				right_leaf_value.append(y_data[i])
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
		#node feature
		# left_feature = np.matrix([])
		# right_feature = np.matrix([])
		# left_feature_f = np.matrix([])
		# right_feature_f = np.matrix([])
		# left_feature_temp = np.matrix([])
		# right_feature_temp = np.matrix([])
		#node value
		# left_value = np.array([])
		# right_value = np.array([])
		# left_value_f = np.array([])
		# right_value_f = np.array([])
		# left_value_temp = np.array([])
		# right_value_temp = np.array([])
		#others
		tree_clp, tree_clm, tree_crp, tree_crm = 0, 0, 0, 0
		feature_temp, feature, real_feature = 0, 0, 0
		#Calculate the number of root node for result 3 and 5
		# temp = 0
		# for i in range(0, len(root_node[:])-1):
		# 	if root_node[i] == 1:
		# 		temp += 1
		# cp = temp
		# cm = len(root_node[:])-temp
		unique, count = np.unique(root_node, return_counts=True)
		cp = count[1]
		cm = count[0]
		#Justify to split or not
		if cp == 0 or cm == 0:
			return 0
		else:
		#Calculate gini-index and benefit
			for j in range(len(root_feature[0])):
				feature_temp = root_feature[:,j]
				# pdb.set_trace()
				for k in range(len(root_node)):
					# print k
					T_temp = feature_temp[k]

					left = root_node[feature_temp <= T_temp]

					right = root_node[feature_temp > T_temp]

					tree_clp, tree_clm = self.count_y(left)
					tree_crp, tree_crm = self.count_y(right)

					# print "here", str(feature_temp[k])
					# pdb.set_trace()
					# start = datetime.datetime.now()
					# tree_clp, tree_clm, tree_crp, tree_crm = self.split_leaf(root_node, root_feature, T_temp, j, tree_clp, tree_clm, tree_crp, tree_crm)
					# print "time spent on node", datetime.datetime.now() - start
					# print tree_clp, tree_clm, tree_crp, tree_crm
					# print len(left), len(right)
					gini_temp = self.gini_benefit(float(cp), float(cm), float(tree_clp), float(tree_clm), float(tree_crp), float(tree_crm))
					# print gini_temp
					if gini_temp > gini_f:
						gini_f= gini_temp
						# left_node_f = left_node_temp
						# right_node_f = left_node_temp
						T_f = T_temp
						feature = feature_temp
				if gini_f > gini:
					gini = gini_f
					# left_node = left_node_f
					# right_node = right_node_f
					T = T_f
					real_feature = feature
		return T, real_feature

  #Make a decision tree
	def build_tree(self):
		level = 0
		time_now = datetime.datetime.now()
		# print "lol"
		self.root = self.find_tree(self.y_train,self.x_train,level)
		print "Time taken to build a tree", datetime.datetime.now() - time_now

	def find_tree(self, y_data, x_data, level, max_depth=20):
		# left_feature, right_feature, left_value, right_value, T, real_feature = None, None, None, None, 0,0
		start = datetime.datetime.now()
		T, feature = self.make_node(y_data, x_data)
		print "time spent on node", datetime.datetime.now() - start
		# pdb.set_trace()
		node = Node(feature, T, y_data)
		left_x, left_y, right_x, right_y = self.partition(y_data, x_data, T, feature)
		print "current level",level
		if level <= max_depth:
			if len(left_y) > 0:
				node.left_child = self.find_tree(left_y, left_x, level+1)
			if len(right_y) > 0:
				node.right_child = self.find_tree(right_y, right_x, level+1)
		return node




if __name__ == '__main__':
	DT = DecisionTree("pa3_train_reduced.csv", "pa3_valid_reduced.csv") 
	DT.build_tree()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
          
        