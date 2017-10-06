import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

row = 13 
data = pd.read_csv('data/data.csv', header=None)[0:row]
print data

# create a dic where the key is a number [1-25] and the value is how many times it was drawn considering all the data selected
# it will be used only if we have duplicated number below
dic = {}
for i in range(0,data.shape[1]):
	for j in range(0,data.shape[0]):
		num = data.as_matrix()[j][i]
		if dic.get(num):
			dic[num] += 1
		else:
			dic[num] = 1

# get the mode of each column and mark on the card
card = []
for i in range(0,15):
	val = int(data.mode().loc[0][i])
	# if it's duplicated, get the most commom of all 
	while (val in card):
		val = max(dic, key=dic.get)
		dic.pop(val)
		# mark the card with the number
	card.insert(i,val)
print card
