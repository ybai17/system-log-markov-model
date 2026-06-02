import csv

import numpy as np
import numpy.linalg as la

DIM = 1299

# Prior Condition: p(s0)
pi = np.zeros((DIM))
# First state Condtion: p(s0, s1)
S0_1 = np.zeros((DIM, DIM))
# Second state Condition: p(s0, s1, s2)
S0_1_2 = np.zeros((DIM, DIM, DIM))

# Conditional Probability p(s1 | s0)
# Indexing is as [s0][s1]
S_1C0 = np.zeros((DIM, DIM))
# Conditional Probability p(s2 | s0, s1)
# Indexing is as [(s0*DIM) + s1][s2]
S_2C01 = np.zeros((DIM ** 2, DIM))

with open('model/S0.csv') as ifp:
  reader = csv.DictReader(ifp)
  for row in reader:
    pi[int(row['state_0'])] += float(row['probability'])

with open('model/S0S1.csv') as ifp:
  reader = csv.DictReader(ifp)
  for row in reader:
    S0_1[int(row['state_0'])][int(row['state_1'])] += float(row['probability'])

with open('model/S0S1S2.csv') as ifp:
  reader = csv.DictReader(ifp)
  for row in reader:
    S0_1_2[int(row['state_0'])][int(row['state_1'])][int(row['state_2'])] += float(row['probability'])

for i in range(DIM):
  if pi[i] != 0:
    for j in range(DIM):
      if S0_1[i][j] != 0:
        S_1C0[i][j] = S0_1[i][j] / pi[i]
        for k in range(DIM):
          S_2C01[i*DIM+j][k] = S0_1_2[i][j][k] / S0_1[i][j]

def one_state_index(s0):
  return s0

def two_state_index(s0, s1):
  return s0 * DIM + s1

def maxlikelihood_activation(s, T):
  D = T[s]
  mi = 0
  for i in range(1, len(D)):
    mi = i if D[i] > D[mi] else mi
  return (mi, D)