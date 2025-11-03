# %%
import numpy as np
import numba as nb
from typing import Iterable, Union, Literal
from bsplyne import BSpline, MultiPatchBSplineConnectivity
from pyxel import Volume
from C1_triplets import get_C1_triplets


def find_displ(
    bsplines: Iterable[BSpline], 
    ctrl_pts: np.ndarray, 
    connectivity: MultiPatchBSplineConnectivity, 
    image: Union[str, Volume, np.ndarray], 
    C1_inds: Union[None, Literal['auto', 'none', 'all'], Iterable[int]]=None, 
    regul_weight: Union[None, float]=None
    ) -> dict:
    
    A, B, C = get_C1_triplets(ctrl_pts, connectivity, C1_inds)
    

# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne import BSpline, MultiPatchBSplineConnectivity

# Degrés et noeuds (identiques pour les deux surfaces)
degrees = [2, 2]
knots = [np.array([0, 0, 0, 1, 1, 1], dtype='float'),
         np.array([0, 0, 0, 1, 1, 1], dtype='float')]

# Création des surfaces B-spline
spline1 = BSpline(degrees, knots)
spline2 = BSpline(degrees, knots)

# Points de contrôle pour la première surface
ctrl_pts1 = np.array([[[ 0.        ,  0.        ,  0.        ],
                       [ 0.52241974,  0.33222979,  0.33012003],
                       [ 0.89672573,  0.97123969,  1.15642331]],
               
                      [[ 0.        ,  0.5       ,  1.        ],
                       [-0.10238268,  0.60889525,  0.87341315],
                       [ 0.14806991,  0.51120647,  1.19544912]],
               
                      [[ 0.        ,  0.        ,  0.        ],
                       [ 0.08212662,  0.18876481,  0.01672326],
                       [ 0.37906962,  0.6309376 ,  0.49510159]]])

# Points de contrôle pour la deuxième surface
ctrl_pts2 = ctrl_pts1.copy()
ctrl_pts2[0] *= -1
ctrl_pts2 = spline2.knotInsertion(ctrl_pts2, [1, 0])

ctrl_pts1 = spline1.knotInsertion(ctrl_pts1, [0, 2])
ctrl_pts2 = spline2.knotInsertion(ctrl_pts2, [0, 2])

# Affichage
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
spline1.plotMPL(ctrl_pts1, ax=ax)
spline2.plotMPL(ctrl_pts2, ax=ax)
plt.show()

conn = MultiPatchBSplineConnectivity.from_separated_ctrlPts([ctrl_pts1, ctrl_pts2])

    
from C1_triplets import get_all_triplets
def make_C1_couples(
    connectivity: MultiPatchBSplineConnectivity, 
    mode: Union[Literal['auto', 'none', 'all'], None]=None
    ) -> tuple[np.ndarray[np.integer], np.ndarray[np.integer], np.ndarray[np.integer]]:
    
    if mode is None:
        mode = 'auto'
    
    if mode=='none':
        A = np.empty(0, dtype='int')
        B = np.empty(0, dtype='int')
        C = np.empty(0, dtype='int')
        return A, B, C
    
    if mode=='all':
        A, B, C = get_all_triplets(conn.unique_nodes_inds, conn.shape_by_patch)
        return A, B, C
    
    if mode=='auto':
        ABC = get_all_triplets(conn.unique_nodes_inds, conn.shape_by_patch)
        a, b, c = ctrl_pts[:, ABC].transpose(1, 0, 2)
        to_keep = np.isclose(b - a, c - b).all(axis=0)
        A, B, C = ABC[:, to_keep]
        return A, B, C
    

# B_mask_unpacked = conn.get_duplicate_unpacked_nodes_mask()
# B, = conn.pack(B_mask_unpacked, method='first').nonzero()
# numbered_mask_unpacked = -1*np.ones(B_mask_unpacked.shape, dtype='int')
# numbered_mask_unpacked[B_mask_unpacked] = np.arange(B.size)
# numbered_mask_separated = conn.separate(numbered_mask_unpacked)
# A = []
# C = []

make_C1_couples(conn)

shape_by_patch = np.array([[3, 3], [4, 3]], dtype='int')
unpacked_inds = np.array([0, 1, 2, 9, 10, 11], dtype='int')
# %%
