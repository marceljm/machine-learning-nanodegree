import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TEIMOSINHA = 12 

# R$ 2,00
PRICE = 2.0

# test all window sizes between 1 and myRange to try to find out the best one
# DON'T use this number. Just get the perfect windows size and run the other script
MAX_WINDOW = 20

#ignore last N results
ignore = 1 

for row in range (1, MAX_WINDOW):
    # read CSV
    data = pd.read_csv('data/data.csv', header=None)[TEIMOSINHA+ignore:TEIMOSINHA+row+ignore]

    # get the mode of each column and mark on the card
    card = []
    for i in range(0,15): 
        val = int(data.mode().loc[0][i])
        # if it's duplicated, get the next one, but increasing 2, to keep the balance between evens and odds
        while (val in card):
            val+=2
            if val>25:
                val=1
        # mark the card with the number
        card.insert(i,val)
    #print card
   
    # get the last TEIMOSINHA results and compute how many bets your card would win
    # data is a array of results
    # i is each result
    # j is each number of the result
    # dic example: [8:1, 9:2, 10:4, 11:7, 12:2, 13:1, 15:1]
    dic = {}
    data = np.array(pd.read_csv('data/data.csv', header=None, nrows=TEIMOSINHA))
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
    print round(total / (TEIMOSINHA * PRICE), 2), ' ', row, dic, card
    


