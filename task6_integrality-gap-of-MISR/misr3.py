#!/usr/bin/env python3
import gurobipy as gp
from gurobipy import GRB

# ----------------------------------------------------------------------
# given arrays (length 16, each label 1..8 appears twice)
arrH = [3,8,6,8,4,5,6,1,1,2,7,3,2,4,5,7]
arrV = [4,8,7,1,6,4,6,3,3,8,2,7,5,5,1,2]


n_rect = 8
GRID   = 16                   # indices 0..15 → cells 1..16 in description

# ----------------------------------------------------------------------
# build rectangles: (x1,x2) × (y1,y2)  with x2,y2 one-past-last
rects = []
for rid in range(1, n_rect+1):
    xs = [i for i,v in enumerate(arrH) if v == rid]
    ys = [i for i,v in enumerate(arrV) if v == rid]
    x1, x2 = min(xs), max(xs)+1      # one-past-last
    y1, y2 = min(ys), max(ys)+1
    rects.append(((x1,x2), (y1,y2)))

# ----------------------------------------------------------------------
# helper: strict interior overlap?
def overlap_1d(a1,a2,b1,b2):
    return (a1 < b2) and (b1 < a2)    # open intervals
# ----------------------------------------------------------------------
# MILP for maximum interior-disjoint subset
m_ilp = gp.Model("mis")
m_ilp.setParam("OutputFlag",0)
x = m_ilp.addVars(n_rect, vtype=GRB.BINARY)
m_ilp.setObjective(gp.quicksum(x[i] for i in range(n_rect)), GRB.MAXIMIZE)

for i in range(n_rect):
    for j in range(i+1, n_rect):
        (x1,x2),(y1,y2) = rects[i]
        (u1,u2),(v1,v2) = rects[j]
        if overlap_1d(x1,x2,u1,u2) and overlap_1d(y1,y2,v1,v2):
            m_ilp.addConstr(x[i] + x[j] <= 1)

m_ilp.optimize()
ilp_val = m_ilp.ObjVal

# ----------------------------------------------------------------------
# LP relaxation
m_lp = gp.Model("misLP")
m_lp.setParam("OutputFlag",0)
xl = m_lp.addVars(n_rect, lb=0, ub=1, vtype=GRB.CONTINUOUS)
m_lp.setObjective(gp.quicksum(xl[i] for i in range(n_rect)), GRB.MAXIMIZE)

for i in range(n_rect):
    for j in range(i+1, n_rect):
        (x1,x2),(y1,y2) = rects[i]
        (u1,u2),(v1,v2) = rects[j]
        if overlap_1d(x1,x2,u1,u2) and overlap_1d(y1,y2,v1,v2):
            m_lp.addConstr(xl[i] + xl[j] <= 1)

m_lp.optimize()
lp_val = m_lp.ObjVal

for v in m_ilp.getVars():
    print(f"{v.VarName} = {v.X}")

for v in m_lp.getVars():
    print(f"{v.VarName} = {v.X}")

# ----------------------------------------------------------------------
print("rectangles (1-based coordinates):")
for idx,((x1,x2),(y1,y2)) in enumerate(rects,1):
    print(f"  {idx}: [{x1+1},{x2}] × [{y1+1},{y2}]")

print("\nILP optimum :", ilp_val)
print("LP  optimum :", lp_val)
print("LP / ILP    :", lp_val / ilp_val if ilp_val else 0.0)
