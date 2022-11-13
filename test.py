
def evidence_updating(x, alpha):
    new_x = (1-alpha)*x/((1-alpha)*x + alpha*(1-x))
    return new_x

alpha =0.1

x = 0.5
i = 0
while x != 1.0:
    print(f"{i} : {x}")
    i += 1
    x = evidence_updating(x, alpha)

