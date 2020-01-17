import numpy as np
import random as rnd
import os
import json
import sys

out_name  = sys.argv[1].split('.')[0]

in_file   = np.genfromtxt('./data/{}'.format(sys.argv[1]))
x         = in_file[:,0:-1]

n_pts     = len(x[:,0])
n_ws      = len(x[0,:])+1

X         = np.ones([n_pts,n_ws])
X[:,1:]   = x
Y         = in_file[:,-1]


jfile     = open('./data/{}'.format(sys.argv[2]))
json_file = json.load(jfile)
alpha     = json_file['learning rate']
n_iter    = json_file['num iter']


#ANALYTIC SOLUTION
temp      = np.linalg.inv(np.dot(X.T,X))
W_star_1  = np.dot(temp, X.T).dot(Y)


#GRADIENT DESCENT
W_init = np.ones(n_ws)

def update_rule(w):
    j = rnd.randint(0,n_pts-1)
    return w + alpha*(Y[j]-np.dot(w, X[j,:]))*X[j,:]

def walker(w):
    for i in range(n_iter):
        w = update_rule(w)
    
    return w


W_star_2 = walker(W_init)

output_file = open('./data/{}.out'.format(out_name),'w')
for i in range(0,n_ws):
    output_file.write('%.4G \n' % W_star_1[i])

output_file.write('\n')

for i in range(0,n_ws):
    output_file.write('%.4G \n' % W_star_2[i])
