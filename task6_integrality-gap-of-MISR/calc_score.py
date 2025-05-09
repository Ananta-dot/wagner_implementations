import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Hardcoded rectangles with pairwise overlaps (corner overlaps)
rectangles = [
    ((0, 2), (0, 1)),   # R1
    ((2, 3), (0, 2)),   # R2
    ((1, 3), (2, 3)),   # R3
    ((0, 1), (1, 3)),   # R4
    ((1, 2), (1, 2)),   # R5 (center rectangle overlaps corners only)
]

# Plot rectangles for visual verification
def plot_rectangles(rectangles):
    fig, ax = plt.subplots()
    for idx, ((x1,x2),(y1,y2)) in enumerate(rectangles):
        ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='blue'))
        ax.text((x1+x2)/2, (y1+y2)/2, f"R{idx+1}", color='red')
    plt.xlim(-1,4)
    plt.ylim(-1,4)
    ax.set_aspect('equal')
    plt.title("Pairwise Corner Overlap Rectangles")
    plt.show()

plot_rectangles(rectangles)

# Compute integrality gap (LP vs ILP)
def integrality_gap(rectangles):
    constraints = []
    for x in range(-1, 5):
        for y in range(-1, 5):
            covered = [i for i, r in enumerate(rectangles)
                       if r[0][0]<=x<=r[0][1] and r[1][0]<=y<=r[1][1]]
            if len(covered) >= 2:
                constraints.append(covered)

    # LP
    lp_model = gp.Model()
    lp_model.setParam('OutputFlag', 0)
    x_lp = lp_model.addVars(len(rectangles), lb=0, ub=1)
    lp_model.setObjective(x_lp.sum(), GRB.MAXIMIZE)
    for c in constraints:
        lp_model.addConstr(sum(x_lp[i] for i in c) <= 1)
    lp_model.optimize()

    # ILP
    ilp_model = gp.Model()
    ilp_model.setParam('OutputFlag', 0)
    x_ilp = ilp_model.addVars(len(rectangles), vtype=GRB.BINARY)
    ilp_model.setObjective(x_ilp.sum(), GRB.MAXIMIZE)
    for c in constraints:
        ilp_model.addConstr(sum(x_ilp[i] for i in c) <= 1)
    ilp_model.optimize()

    lp_val = lp_model.objVal
    ilp_val = ilp_model.objVal
    ratio = lp_val / ilp_val if ilp_val else 0

    print(f"LP solution = {lp_val}")
    print(f"ILP solution = {ilp_val}")
    print(f"Integrality gap (LP/ILP) = {ratio:.3f}")

integrality_gap(rectangles)
