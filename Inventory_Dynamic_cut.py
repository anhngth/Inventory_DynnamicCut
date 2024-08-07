import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# Assuming ED1 is a list of expected demands for each period
ED1 = [
    [166, 209, 398, 497, 577, 701, 735, 774, 859, 905, 1040, 1124, 1201, 1257, 1302, 1446, 1627, 1698, 1859, 1922, 2002],
    [209, 43, 232, 331, 411, 535, 569, 608, 693, 739, 874, 958, 1035, 1091, 1136, 1280, 1461, 1532, 1693, 1756, 1836]
]

# Parameters
N = 20  # replace with actual value
m = 21 #replenishment cycle
M = 9999999  # replace with actual value
h = 1  # holding cost
p = 2  # backorder cost
K = 225  # fixed setup cost

# Initialize the model
model = LpProblem("Optimization Model", LpMinimize)

# Decision variables
xij = LpVariable.dicts("xij", [(i, j) for i in range(m) for j in range(i, m + 1)], cat='Binary')
qij = LpVariable.dicts("qij", [(i, j) for i in range(m) for j in range(i, m + 1)], lowBound=0)
H = LpVariable.dicts("H", [(i, j, t) for i in range(m) for j in range(i, m + 1) for t in range(i, j)], lowBound=0)

# Objective Function
model += lpSum([K * xij[(i,j)] + lpSum([(h * (qij[(i,j)] - ED1[1][t] * xij[(i,j)]) + (h + p) * H[(i, j, t)])
               for t in range(i, j - 1)])
               for i in range(1, N+1) for j in range(i + 1, N + 2)])


# Constraint (3): Replenishment cycle logic
for t in range(2, N + 1):
   model += lpSum([xij[i, t] for i in range(1, t)]) == lpSum([xij[t, j] for j in range(t + 1, N + 2)])

# Constraint (4) and (5): First and last replenishment cycle logic
model += lpSum([xij[1, j] for j in range(2, N + 2)]) == 1
model += lpSum([xij[i, N + 1] for i in range(N+1)]) == 1


# Constraint (6): Expected cumulative order quantity
for i in range(1, N + 1):
   for j in range(i + 1, N + 2):
       model += qij[i, j] <= M*xij[i,j]


# Constraint (7): Expected Replenishment Quantity
for t in range(2, N + 1):
    model += lpSum([qij[i, t] for i in range(1,t-1)]) <= lpSum([qij[t, j] for j in range(t + 1, N + 2)])
# Constraint (8): Lower Bound to Exact Loss Function Value
for i in range(1, N + 1):
   for j in range(i + 1, N + 2):
       for t in range(i, j):
           model += H[i, j, t] >= -(qij[i, j] - ED1[1][t] * xij[i, j])


# Constraint (9): Variable domains
for i in range(1, N + 1):
   for j in range(i + 1, N + 2):
       for t in range(i, j):
           model += H[i, j, t] >= 0
           model += qij[i, j] >= 0

# Solve the model
model.solve()


# Export H to a text file
with open("H_output.txt", "w") as file:
    for i in range(m):
        for j in range(i, m + 1):
            for t in range(i, j):
                variable_name = f"H({i},{j},{t})"
                variable_value = H[(i, j, t)].varValue
                file.write(f"{variable_name} = {variable_value}\n")