import docplex.mp.constr
from docplex.mp.model import Model

mdl = Model("model")
x1 = mdl.continuous_var(name="x1")
y1 = mdl.integer_var(name="y1")
y2 = mdl.integer_var(name="y2")
y3 = mdl.integer_var(name="y3")

mdl.add_constraint(y1 + y2 - y3 + 3 * x1 <= 7)
mdl.add_constraint(y2 + 3 * y3 - x1 <= 5)
mdl.add_constraint(- 3 * y1 - x1 <= -2)

mdl.maximize(-y1 + 2 * y2 + y3 + 2 * x1)

print(mdl.export_to_string())

solution = mdl.solve(log_output=True)
solution.display()
