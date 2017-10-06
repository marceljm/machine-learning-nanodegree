import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

money = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0}

TEIMOSINHA = 12 

# R$ 2,00
PRICE = 2.0

# test all window sizes between 1 and myRange to try to find out the best one
# DON'T use this number. Just get the perfect windows size and run the other script
MIN_WINDOW = 1 
MAX_WINDOW = 20 

#ignore last N results // 1 ou 2
ignore = 0 

MAX_TEST = 15 

for n in range (0, MAX_TEST):
	for row in range (MIN_WINDOW, MAX_WINDOW):
		# read CSV
		data = pd.read_csv('data/data.csv', header=None)[TEIMOSINHA+ignore+n:TEIMOSINHA+row+ignore+n]

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
		#print card

		# get the last TEIMOSINHA results and compute how many bets your card would win
		# data is a array of results
		# i is each result
		# j is each number of the result
		# dic example: [8:1, 9:2, 10:4, 11:7, 12:2, 13:1, 15:1]
		dic = {}
		data = np.array(pd.read_csv('data/data.csv', header=None, nrows=TEIMOSINHA, skiprows=n))
		for i in data:
			count = 0
			for j in i:
				if j in card:
					count+=1
			if count in dic.keys():
				dic[count] += 1
			else:
				dic[count]=1
		#print dic

		# computer the money lost and earned
		total = 0
		for i in dic.keys():
			if i == 11:
				total += 4 * dic[i]
			elif i == 12:
				total += 8 * dic[i]
			elif i == 13:
				total += 20 * dic[i]
			elif i == 14:
				total += 1682 * dic[i]
			elif i == 15:
				total += 962407 * dic[i]
		#print round(total / (TEIMOSINHA * PRICE), 2),';',row,'-',n,dic, card
		money[row]+=total / (TEIMOSINHA * PRICE)
for i in money:
	print round(money[i]/MAX_TEST,2), i

