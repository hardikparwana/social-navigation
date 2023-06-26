# (Conformal Prediction) Computing d values

# Get d values

import ndjson
import pandas as pd
import os
import math
import numpy as np
from numpy import linalg as LA

H = 20
o_len = 20
p_len = 20

priv = []
pred = []
pv = "test_private_"
pd = "test_pred_"

d = "d"
x_cha = "x"
y_cha = "y"
x_l = "x_LSTM"
y_l = "y_LSTM"

d_list = []
x_list = []
y_list = []
x_l_list = []
y_l_list = []

for i in range(p_len):
    priv.append(pv + str(o_len+i))
    pred.append(pd + str(o_len+i) + "/")
    d_list.append(d+str(i+o_len))
    x_list.append(x_cha+str(i+o_len))
    y_list.append(y_cha+str(i+o_len))
    x_l_list.append(x_l+str(i+o_len))
    y_l_list.append(y_l+str(i+o_len))


import pickle

# save_path = "aaa/maxNorm/"
save_path = "/home/hardik/Research/social-navigation/conformal_prediction/maxNorm/"


fid = open(save_path + "d_value.pkl", 'rb')
d_value= pickle.load(fid)
fid.close()

fid = open(save_path + "x_value.pkl", 'rb')
x_value = pickle.load(fid)
fid.close()


fid = open(save_path + "y_value.pkl", 'rb')
y_value = pickle.load(fid)
fid.close()



fid = open(save_path + "x_l_value.pkl", 'rb')
x_l_value = pickle.load(fid)
fid.close()

fid = open(save_path + "y_l_value.pkl", 'rb')
y_l_value = pickle.load(fid)
fid.close()

print(f"x_l_value:{x_l_value}")

# pred_lstm = 'lstm_goals_social_None_20_2000_modes1/'
# path = "/data2/mcleav/conformalRNNs/icra_2022/code/Trajnet_test/trajnetplusplusbaselines/DATA_BLOCK/synth_data/"
# path = "/home/hardik/Research/social-navigation/conformal_prediction"
path = "/home/hardik/Research/social-navigation/conformal_prediction/dataForShuo/lstm_goals_social_None_20_2000_modes1"

# (Conformal Prediction) Find an ID, Ped Numbers

# Pedestrian 1
Spec_ID_1 = []

# Modify numbers
x_H10_1 = []
y_H10_1 = []
d_20_1 = []

# os.chdir(path + str(pred[0])+ str(pred_lstm))
os.chdir(path)
with open('orca_three_synth.ndjson', 'r') as f:
    predict_data=ndjson.load(f)

s_point = 104533
p_point = 3617
Spec_ID_1.append(1997)

x_H10_1.append(x_l_value[x_l_list[0]]['ID: '+str(Spec_ID_1[0])]['Ped 1'])
y_H10_1.append(y_l_value[y_l_list[0]]['ID: '+str(Spec_ID_1[0])]['Ped 1'])
d_20_1.append(d_value[d_list[0]])

for q in range(1, p_len):
    # os.chdir(path + str(pred[q])+ str(pred_lstm))
    os.chdir(path)
    with open('orca_three_synth.ndjson', 'r') as f:
        predict_data=ndjson.load(f)
    # print(f"predict_data:{predict_data}")
    for i in range(len(predict_data)):
        if list(predict_data[i].keys())[0] == 'scene':
            if list(predict_data[i].values())[0].get("s") == s_point:
                if list(predict_data[i].values())[0].get("p") == p_point:
                    Spec_ID_1.append(predict_data[i]['scene']['id'])
    
    x_H10_1.append(x_l_value[x_l_list[q]]['ID: '+str(Spec_ID_1[q])]['Ped 1'])
    y_H10_1.append(y_l_value[y_l_list[q]]['ID: '+str(Spec_ID_1[q])]['Ped 1'])
    d_20_1.append(d_value[d_list[q]])
    
original_x_1 = x_value[x_list[0]]['ID: '+str(1997)]['Ped 1']
original_y_1 = y_value[y_list[0]]['ID: '+str(1997)]['Ped 1']

# %cd /Users/joey