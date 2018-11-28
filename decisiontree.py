# !/usr/bin/env python

import numpy as np
import csv
import sys, os, pdb
from process_data import *

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
  def gini_benefit(cp,cm,clp,clm,crp,crm):
    if clp+clm==0 or crp+crm==0:
      return 0
    pUl = (2*clp*clm/((clp+clm)*(cp+cm)))
    pUr = (2*crp*crm/((crp+crm)*(cp+cm)))
    Ua =  (2*cp*cm/(cp+cm)**2)
    return (Ua-pUl-pUr)
  
  def feature_split(self):
    #initialize the parameters
    gini = 0
    T = 0
    feature = 0
    fclp, fclm, fcrp, fcrm = 0, 0, 0, 0
    
    #Calculate the number of result 3 and 5
    temp = 0
    for i in range(0, len(self.y_train[:])-1):
      if self.y_train[i] == 1:
        temp += 1
    cp = temp
    cm = len(self.y_train[:])-temp
    #Calculate gini-index and benefit
    for j in range(0, len(self.x_train[0][:])-1):
      for k in range(0, len(self.x_train[:][0])-1):
        T = self.x_train[k][j]
        if 
        