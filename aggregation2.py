# python aggregation2.py <number of files> <submission file> [<submission file> ...] <output file>
# python aggregation2.py 5 predicted_1.csv predicted_2.csv predicted_3.csv predicted_4.csv predicted_5.csv predicted_a.csv
import numpy as np
import pandas as pd
import sys

nb_csv = int(sys.argv[1])


# predicted
prob = []
id = None
for i in range(2,2+nb_csv):
    predicted = pd.read_csv(sys.argv[i], header=None)
    print('predicted({0[0]},{0[1]})'.format(predicted.shape))
    print (predicted.head())

    prob.append(predicted.iloc[:,1:].values.astype(np.float))
    if id is None:
        id = predicted[[0]].values.ravel().astype(np.str)

validation_predicted_aggregation_csv = sys.argv[2+nb_csv]

# save results
lines=[]
for i in range(0,id.shape[0]):
    lb = np.array([])
    pb = np.array([])
    count = np.zeros(10)
    for j in range(0,nb_csv):
        l = np.argmax(prob[j][i])
        lb = np.append(lb, l)
        pb = np.append(pb, np.max(prob[j][i]))
        count[l] = count[l] + 1
    # get max pb
    lmax = np.argmax(count)
    keep = []
    for j in range(0,nb_csv):
        if lb[j] == lmax:
            keep.append(j)
    pb2 = pb[keep]
    if np.max(count) != nb_csv:
        pmax = keep[np.argmax(pb2)]
        print('i => {4}, id => {3}, count => {0}, lb => {1}, pb => {2}').format(count, lb, pb, id[i], i)
        print('keep => {0}, lmax => {1}, pmax => {2}, pb2 => {3}').format(keep, lmax, pmax, pb2)
    else:
        pmax = keep[np.argmax(pb2)]
    lines.append([id[i]]+map(str,prob[pmax][i]))

with open(validation_predicted_aggregation_csv, 'wb') as f:
    for l in lines: f.write(','.join(l)+'\n')






