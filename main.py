# ================================
#       ESEMPIO DI UTILIZZO
# ================================
from utils import calcola_jacobiana, dh_to_transformation_matrices

# Variabili simboliche
q1, q2, q3 = sp.symbols('q1 q2 q3')

# Parametri DH [d, theta, a, alpha]
dh_params = [
    [sp.pi/2,       0,   0,           0],
    [sp.pi/2,       q2,  0,  -sp.pi/2],
    [0,             q3,  0,           0]
]

# Tipologia dei giunti
joint_types = ['R', 'P', 'R']

# Trasformazione base → mondo
T_0 = sp.Matrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

# Calcolo trasformazioni
dh_matrices = dh_to_transformation_matrices(dh_params)

# Trasformazione base → end-effector
T_base_to_EE = sp.eye(4)
for mat in dh_matrices:
    T_base_to_EE = T_base_to_EE @ mat

T_total = T_0 @ T_base_to_EE

# Calcolo Jacobiana
_, J = calcola_jacobiana(dh_matrices, joint_types)

# ================================
#           OUTPUT
# ================================
print("=== T_0 (World to Base) ===")
sp.pprint(sp.simplify(T_0), use_unicode=True)

for i, T in enumerate(dh_matrices):
    print(f"\n=== T_{i} to T_{i+1} ===")
    sp.pprint(sp.simplify(T), use_unicode=True)

print("\n=== T_Base to EE ===")
sp.pprint(sp.simplify(T_base_to_EE), use_unicode=True)

print("\n=== T_World to EE ===")
sp.pprint(sp.simplify(T_total), use_unicode=True)

print("\n=== Jacobiana Geometrica ===")
sp.pprint(sp.simplify(J), use_unicode=True)