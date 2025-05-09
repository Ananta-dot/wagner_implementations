import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt
import random
import math

def decode_squares_from_bits(bit_array, num_squares, bits_per_coord, square_width, coord_max):
    """
    Interpret 'bit_array' as (x,y) pairs, each with bits_per_coord bits. 
    Return list of squares: (x1,x2,y1,y2), each with side square_width 
    placed in [0..coord_max].
    """
    BITS_PER_SQUARE = 2 * bits_per_coord
    DECISIONS = num_squares * BITS_PER_SQUARE
    
    if len(bit_array) != DECISIONS:
        raise ValueError(f"Bit array length mismatch: expected={DECISIONS}, got={len(bit_array)}")

    squares = []
    max_int = (1 << bits_per_coord) - 1  # for the scaling
    for i in range(num_squares):
        start_idx = i * BITS_PER_SQUARE
        x_bits = bit_array[start_idx : start_idx + bits_per_coord]
        y_bits = bit_array[start_idx + bits_per_coord : start_idx + BITS_PER_SQUARE]
        
        x_val = int("".join(str(b) for b in x_bits), 2)
        y_val = int("".join(str(b) for b in y_bits), 2)
        
        if max_int>0:
            x_coord = int((x_val / max_int) * max(0, coord_max - square_width))
            y_coord = int((y_val / max_int) * max(0, coord_max - square_width))
        else:
            x_coord = 0
            y_coord = 0
        
        x1, x2 = x_coord, x_coord + square_width
        y1, y2 = y_coord, y_coord + square_width
        squares.append((x1, x2, y1, y2))
    return squares

def run_gurobi_stabbing(squares):
    """
    squares: list of (x1, x2, y1, y2)
    Returns:
       ILP_val, ILP_sol_V, ILP_sol_H, LP_val, LP_sol_V, LP_sol_H
    where ILP_sol_V and ILP_sol_H are dictionaries mapping each candidate x- or y-coordinate 
    to its ILP (0/1) value, and LP_sol_V and LP_sol_H are dictionaries mapping each candidate coordinate 
    to its LP fractional value.
    """
    # Gather candidate lines: here we take both left/right boundaries (x1 and x2) and top/bottom (y1 and y2)
    xcoords = set()
    ycoords = set()
    for (x1, x2, y1, y2) in squares:
        xcoords.update([x1, x2])   # add both vertical boundaries
        ycoords.update([y1, y2])
    xcoords = sorted(list(xcoords))
    ycoords = sorted(list(ycoords))

    # Build ILP model
    model_ilp = gp.Model("stab_ILP")
    model_ilp.Params.OutputFlag = 0
    V = model_ilp.addVars(len(xcoords), vtype=GRB.BINARY, name="V")
    H = model_ilp.addVars(len(ycoords), vtype=GRB.BINARY, name="H")
    for (x1, x2, y1, y2) in squares:
        x_idxs = [i for i, xv in enumerate(xcoords) if x1 <= xv <= x2]
        y_idxs = [j for j, yv in enumerate(ycoords) if y1 <= yv <= y2]
        model_ilp.addConstr(gp.quicksum(V[i] for i in x_idxs) +
                            gp.quicksum(H[j] for j in y_idxs) >= 1)
    model_ilp.setObjective(V.sum("*") + H.sum("*"), GRB.MINIMIZE)
    model_ilp.optimize()
    if model_ilp.status != GRB.OPTIMAL:
        return (0, {}, {}, 0, {}, {})

    ILP_val = model_ilp.ObjVal
    ILP_sol_V = {xcoords[i]: V[i].x for i in range(len(xcoords))}
    ILP_sol_H = {ycoords[j]: H[j].x for j in range(len(ycoords))}

    # Build LP relaxation
    model_lp = model_ilp.relax()
    model_lp.Params.OutputFlag = 0
    model_lp.optimize()
    if model_lp.status != GRB.OPTIMAL:
        return (ILP_val, ILP_sol_V, ILP_sol_H, 0, {}, {})

    LP_val = model_lp.ObjVal
    LP_sol_V = {}
    LP_sol_H = {}
    for i in range(len(xcoords)):
        var = model_lp.getVarByName(f"V[{i}]")
        LP_sol_V[xcoords[i]] = var.x
    for j in range(len(ycoords)):
        var = model_lp.getVarByName(f"H[{j}]")
        LP_sol_H[ycoords[j]] = var.x

    return (ILP_val, ILP_sol_V, ILP_sol_H, LP_val, LP_sol_V, LP_sol_H)

def plot_squares_and_lines(squares,
                           ILP_sol_V, ILP_sol_H,
                           LP_sol_V,  LP_sol_H,
                           ILP_val, LP_val):
    ratio = ILP_val / LP_val if LP_val > 1e-9 else 0
    fig, ax = plt.subplots(figsize=(7, 7))
    xs, ys = [], []

    # ---- draw the squares --------------------------------------------------
    for i, (x1, x2, y1, y2) in enumerate(squares):
        rect = plt.Rectangle((x1, y1),
                             x2 - x1, y2 - y1,
                             facecolor=(0.6, 0.6, 0.9, 0.3),
                             edgecolor='black')
        ax.add_patch(rect)
        ax.text((x1+x2)/2, (y1+y2)/2, str(i+1),
                ha='center', va='center', fontsize=8)
        xs.extend([x1, x2]);  ys.extend([y1, y2])

    # ---- dashed grey for *all* candidates ----------------------------------
    for xv in ILP_sol_V: ax.axvline(xv, color='gray', ls='--', lw=1)
    for yv in ILP_sol_H: ax.axhline(yv, color='gray', ls='--', lw=1)

    # ---- solid blue for ILP‑chosen lines -----------------------------------
    for xv, val in ILP_sol_V.items():
        if val > 0.0:
            ax.axvline(xv, color='blue', lw=2)
    for yv, val in ILP_sol_H.items():
        if val > 0.0:
            ax.axhline(yv, color='blue', lw=2)

    # ---- solid red for LP‑only (fractional) lines --------------------------
    for xv, frac in LP_sol_V.items():
        if frac > 1e-6 and ILP_sol_V.get(xv, 0) == 0:
            ax.axvline(xv, color='red', lw=2)
    for yv, frac in LP_sol_H.items():
        if frac > 1e-6 and ILP_sol_H.get(yv, 0) == 0:
            ax.axhline(yv, color='red', lw=2)

    # ---- annotations -------------------------------------------------------
    ax.set_xlim(min(xs)-1, max(xs)+2)
    ax.set_ylim(min(ys)-1, max(ys)+2)
    ax.set_title(f"ILP={ILP_val:.3f}, LP={LP_val:.3f},  ratio={ratio:.3f}")
    ax.set_xlabel("X");  ax.set_ylabel("Y")
    plt.show()


if __name__=="__main__":
    # Example bits from your snippet 

    bits=[0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]

    NUM_SQUARES = 6
    # define COORD_MAX and SQUARE_WIDTH EXACTLY as in your RL script
    COORD_MAX = (NUM_SQUARES + 1) * NUM_SQUARES#10 * NUM_SQUARES 
    SQUARE_WIDTH = NUM_SQUARES#0.2 * COORD_MAX
    COORD_MAX = COORD_MAX - SQUARE_WIDTH 
    # we want them in [0..(COORD_MAX - SQUARE_WIDTH)]
    # bits_per_coord = e.g. 3 or 4, must match your RL 
    bits_per_coord =  3 * math.ceil(math.log2(NUM_SQUARES))

    # decode
    squares = decode_squares_from_bits(bits, 
                                       num_squares=NUM_SQUARES, 
                                       bits_per_coord=bits_per_coord,
                                       square_width=SQUARE_WIDTH, 
                                       coord_max=COORD_MAX)

    # Gurobi
    ILP_val, ILP_sol_V, ILP_sol_H, LP_val, LP_sol_V, LP_sol_H = run_gurobi_stabbing(squares)

    ratio = ILP_val/LP_val if LP_val>1e-9 else 0
    print(f"ILP= {ILP_val}, LP= {LP_val}, ratio= {ratio}")

    # Plot everything
    plot_squares_and_lines(squares, ILP_sol_V, ILP_sol_H, LP_sol_V, LP_sol_H, ILP_val, LP_val)

