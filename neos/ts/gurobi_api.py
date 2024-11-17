from gurobipy import read, GRB
model = read('model.ilp')
model.optimize()

if model.status == GRB.OPTIMAL:
    for v in model.getVars():
        print(f'{v.varName} = {v.x}')
    print(f'Optimal Objective: {model.ObjVal}')
else:
    print("No optimal solution found.")
