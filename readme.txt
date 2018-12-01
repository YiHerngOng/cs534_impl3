Implementation assignment 3

To run part 1 : Simple Decision Tree
$ python decisiontree.py [maximum depth of tree]
for example:
$ python decisiontree.py 20
This will show training accuracy and validation accuracy at each level of tree.

To run part 2
$ python randomforest.py [number of trees] [maximum depth of each tree] [number of random features]
for example:
$ python randomforest.py 25 9 10
This will show training accuracy and validation accuracy of the random forest

To run part 3
$ python adaboost.py [number of loop L] [maximum depth of tree]
for example:
$ python adaboost.py 10 9
This will show training accuracy and validation accuracy of decision tree with adaboost