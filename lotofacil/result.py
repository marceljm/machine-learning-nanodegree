import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

row = 13 
data = pd.read_csv('data/data.csv', header=None)[0:row]
print data
card = []
for i in range(0,15): 
    val = int(data.mode().loc[0][i])
    while (val in card):
        val+=2
        if val==26:
            val=2
        elif val==27:
            val=1
    card.insert(i,val)
print card
