import sympy as sp

# ================================
#       UTILITY MATRICES
# ================================
def transformation_matrix(rotation, translation):
    """Crea una matrice di trasformazione 4x4 da rotazione e traslazione."""
    T = sp.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def dh_matrix(d, theta, a, alpha):
    """Genera una matrice di trasformazione DH simbolica."""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha),  sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0,              sp.sin(alpha),                 sp.cos(alpha),                d],
        [0,              0,                             0,                            1]
    ])

def dh_to_transformation_matrices(dh_params):
    """Genera la lista di matrici DH da una lista di parametri [d, theta, a, alpha]."""
    return [dh_matrix(*params) for params in dh_params]

# ================================
#         JACOBIAN CALC
# ================================
def calcola_jacobiana(transformations, joint_types):
    """Calcola T finale e Jacobiana geometrica (6xN)."""
    n = len(transformations)
    assert len(joint_types) == n, "joint_types deve avere lo stesso numero di elementi delle trasformazioni."

    T_matrices = [sp.eye(4)]
    for i in range(n):
        T_matrices.append(T_matrices[i] * transformations[i])

    Ps, Zs = [], []
    for T in T_matrices:
        R = T[:3, :3]
        p = T[:3, 3]
        z = R[:, 2]
        Ps.append(p)
        Zs.append(z)

    p_n = Ps[-1]
    Jv, Jw = [], []

    for i in range(n):
        zi = Zs[i]
        pi = Ps[i]
        if joint_types[i] == 'R':
            Jv.append(zi.cross(p_n - pi))
            Jw.append(zi)
        elif joint_types[i] == 'P':
            Jv.append(zi)
            Jw.append(sp.Matrix([0, 0, 0]))
        else:
            raise ValueError(f"Tipo di giunto non valido: {joint_types[i]}")

    J = sp.Matrix.vstack(sp.Matrix.hstack(*Jv), sp.Matrix.hstack(*Jw))
    return T_matrices[-1], J

# ================================
#        ROTATION TO w
# ================================
def rotation_to_angular_velocity(R, param_symbol, R_dot=None):
    """Calcola il vettore angolare w da una matrice di rotazione R e simbolo di derivazione."""
    if R_dot is None:
        R_dot = R.applyfunc(lambda x: sp.diff(x, param_symbol))

    S_w = R.T * R_dot
    w = sp.Matrix([S_w[2, 1], S_w[0, 2], S_w[1, 0]])
    return w

# ================================
#          RANK UTILITY
# ================================
def rank_calculation(matrice):
    """Calcola il rango di una matrice simbolica."""
    return matrice.rank()