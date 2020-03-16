# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:04:15 2019

@author: Marija
"""
import matplotlib.pylab as plt

#50units
#lstm
t_loss_1 = [270,212,192,169,153,134,120,110,103,98,95,92,89,87,85,83,81,80,79,78]
v_loss_1 = [318,314,313,309,295,251,212,176,171,163,165,157,161,154,153,149,151,147,141,149]

#gru
t_loss_2 = [335,180,123,109,101,96,92,89,87,85,83,81,79,78,76,75,74,74,73,72]
v_loss_2 = [300,204,184,175,174,170,159,162,156,161,154,159,153,148,148,146,148,148,149,139]

plt.xlabel('Epochs')
plt.ylabel('CTC Loss')

x_axis = list(range(1,21))

plt.plot(x_axis,t_loss_1,'b', label='LSTM-50 training loss')
plt.plot(x_axis,t_loss_2,'g', label='GRU-50 training loss')
plt.plot(x_axis,v_loss_1,'b-.', label='LSTM-50 validation loss')
plt.plot(x_axis,v_loss_2,'g-.', label='GRU-50 validation loss')
plt.legend()
plt.xticks(x_axis)

#plt.show()
plt.savefig('50neurons-loss.png',dpi=600)

#100units

#lstm
t_loss_1 = [307,197,187,193,208,208,207,206,206,204,196,183,176,174,167,161,156,154,149,159]
v_loss_1 = [303,289,249,278,278,278,276,277,279,276,259,287,245,236,225,230,232,212,214,230]

#gru
t_loss_2 = [185,166,107,90,81,75,71,67,65,64,62,61,60,59,58,57,57,56,55,55]
v_loss_2 = [380,194,171,152,148,142,132,127,128,132,127,124,129,134,129,127,137,133,127,130]

plt.xlabel('Epochs')
plt.ylabel('CTC Loss')

x_axis = list(range(1,21))

plt.plot(x_axis,t_loss_1,'b', label='LSTM-100 training loss')
plt.plot(x_axis,t_loss_2,'g', label='GRU-100 training loss')
plt.plot(x_axis,v_loss_1,'b-.', label='LSTM-100 validation loss')
plt.plot(x_axis,v_loss_2,'g-.', label='GRU-100 validation loss')
plt.legend()
plt.xticks(x_axis)

#plt.show()
plt.savefig('100neurons-loss.png',dpi=600)