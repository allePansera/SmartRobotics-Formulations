# ================================
#       ESEMPIO DI UTILIZZO
# ================================
from utils import calcola_jacobiana, dh_to_transformation_matrices, rank_calculation
import sympy as sp


# Variabili simboliche
q3 = sp.symbols('q3')


# Trasformazione A
A = sp.Matrix([
    [0, 1, sp.cos(q3)],
    [1, 0, -sp.sin(q3)],
    [0, 0, -1]
])

rank = rank_calculation(A)

# ================================
#           OUTPUT
# ================================

print("\n=== Rank ===")
sp.pprint(rank, use_unicode=True)
