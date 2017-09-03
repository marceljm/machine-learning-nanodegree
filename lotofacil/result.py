import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

row = 9 
data = pd.read_csv('data/data.csv', header=None)[0:row]
print data
card = []
for i in range(0,15): 
    val = int(data.mode().loc[0][i])
    while (val in card):
        val+=1
        if val>25:
            val=1
    card.insert(i,val)
print card
