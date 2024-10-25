# %% [markdown]
# # Modified Gram-Schmidt (MGS) vs Householder’s Triangularization (HHT)

# %% [markdown]
# ### Comparison of Modified Gram-Schmidt (MGS) vs Householder’s Triangularization (HHT) in a computer assuming each computation is calculated on a computer which rounds all computed results to five digits of relative accuracy.

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

A = np.array(
    [
        [0.70000, 0.70711],
        [0.70001, 0.70711]
    ]
)

I = np.array(
    [
        [1, 0],
        [0, 1]
    ]
)

# %%
def round_to_5(x : Any) -> Any:
    return np.round(x, 5)

def MGS(A : np.array) -> Tuple[np.array, np.array]:
    """
    Returns Q, R for matrix A.
    """
    m,n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    V = np.zeros((m,n))

    for i in range(n):
        V[:,i] = deepcopy(A[:,i])
    for i in range(n):
        R[i,i] = round_to_5(np.linalg.norm(V[:,i],2)) 
        Q[:,i] = round_to_5(V[:,i]/R[i,i])

        for j in range(i+1,n):
            R[i,j] = round_to_5(np.dot(Q[:,i],V[:,j]))
            V[:,j] = round_to_5(V[:,j] - R[i,j]*Q[:,i])

    return Q, R

def Householder(A: np.array) -> Tuple[np.array, np.array]:
    """
    Takes in A and b to return Qtb and R using Householder Triangularization (Reduced).
    """
    m,n = A.shape
    R = deepcopy(A)
    Q = np.eye(m)
    for k in range(n):
        x = deepcopy(R[k:, k])
        e_1 = np.zeros_like(x); e_1[0] = 1
        v_k = round_to_5(np.sign(x[0])* round_to_5(np.linalg.norm(x,2)) * e_1 + x)
        v_k = np.reshape(v_k, (m-k,1))
        norm_v_k = round_to_5(np.linalg.norm(v_k,2))
        v_k = round_to_5(v_k / norm_v_k)
        
        R[k:, k:] = round_to_5(R[k:, k:] -  2 * v_k @ v_k.T @ R[k:, k:])
        Q[:, k:] = round_to_5(Q[:, k:] - 2 * np.outer(np.dot(Q[:, k:], v_k), v_k))

     # Trim Q to be reduced size (m x n)
    Q_reduced = Q[:, :n]
    
    return Q_reduced, R[:n, :]

# %%
Q_MGS, R_MGS = MGS(A)
Q_HHT, R_HHT = Householder(A)
orthogonality_loss_MGS = np.linalg.norm(I - Q_MGS.T @ Q_MGS)
orthogonality_loss_HHT = np.linalg.norm(I - Q_HHT.T @ Q_HHT)

k_A = np.linalg.norm(A, 2)*np.linalg.norm(np.linalg.inv(A), 2)

print(f"Relative condition number of A = {k_A}")
print("=======================================================================")
print(f"Q matrix calculated by MGS :\n {Q_MGS}")
print(f"R matrix calculated by MGS :\n {R_MGS}")
print(f"Loss of orthogonality ||I - Q_MGS.T Q_MGS|| = {orthogonality_loss_MGS}")
print("=======================================================================\n")

print("=======================================================================")
print(f"Q matrix calculated by HHT :\n {Q_HHT}")
print(f"Q matrix calculated by HHT :\n {R_HHT}")
print(f"Loss of orthogonality ||I - Q_HHT.T Q_HHT|| = {orthogonality_loss_HHT}")
print("=======================================================================\n")

# %% [markdown]
# We can see from the results above that Q_MGS computed by MGS has very high loss of orthogonality (0.99999) compared to Q_HHT ( loss =  2.68941e-05.). Therefore Householder algorithm is preferred on low precision machines.
# 
# 
# 

# %%


# %%



