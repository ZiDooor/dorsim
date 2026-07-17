import itertools
import numpy as np
from tqdm import tqdm
from dorsim import (
    BiasedPoulinDecoder,
    CSSCode,
    StabilizerCode,
    concat_code,
)
import matplotlib.pyplot as plt

def get_depolarizing_error(n:int, p:float, batch_size:int|None=None):
    np_rng = np.random.default_rng()
    assert 0<=p<=1
    isone = batch_size is None
    if isone:
        batch_size = 1
    tmp0 = np_rng.choice(4, p=np.array([1-p, p/3, p/3, p/3]), size=(batch_size,n))
    # 0123: IZXY
    ret = np.concatenate([tmp0//2, tmp0%2], axis=1).astype(np.uint8)
    if isone:
        ret = ret[0]
    return ret

def matmul_gf4(np0, np1):
    """Matrix multiplication in GF(4) symplectic representation.

    Computes matrix product respecting symplectic/GF(4) structure where
    matrices are stored as [X part | Z part].

    Args:
        np0: Binary matrix, shape (..., 2n), dtype=np.uint8.
        np1: Binary matrix, shape (2n,) or (..., 2n, k), dtype=np.uint8.

    Returns:
        np.ndarray: Product matrix, preserving symplectic structure.
    """
    assert (np0.dtype==np.uint8) and (np1.dtype==np.uint8)
    assert np0.shape[-1]%2==0
    assert np0.shape[-1] == np1.shape[(0 if (np1.ndim==1) else -2)]
    N0 = np0.shape[-1] // 2
    ind0 = slice(0,N0)
    ind1 = slice(N0,2*N0)
    np0a = np0[...,ind0]
    np0b = np0[...,ind1]
    if np1.ndim==1:
        assert np1.shape[0]
        ret = (np0a @ np1[ind1] + np0b @ np1[ind0]) % 2
    else: #np1.ndim>=2
        ret = (np0a @ np1[...,ind1,:] + np0b @ np1[...,ind0,:]) % 2
    return ret

def steane_concat_capacity():
    code = StabilizerCode.steane()
    code_l2 = concat_code(code, [code, code, code, code, code, code, code])
    code_l3 = concat_code(code, [code_l2, code_l2, code_l2, code_l2, code_l2, code_l2, code_l2])
    code_list = [code, code_l2, code_l3]
    level_list = [1, 2, 3]
    
    num_sample = int(1e5)
    plist = np.linspace(0.1, 0.2, 10)
    batch_size = 512

    num_fail_list = []
    num_total_list = []
    for level_i in level_list:
        code = code_list[level_i-1]
        stab_list = code.stabilizers
        logical_list = np.concatenate([code.logical_x, code.logical_z], axis=0)
        decoder = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)
        for p in plist:
            num_fail = 0
            num_total = 0
            decoder.set_error_model(p/3, p/3, p/3)
            for _ in tqdm(range(num_sample//batch_size), desc=f'level {level_i} p={p:.3f}'):
                error = get_depolarizing_error(code.n, p, batch_size)
                syndrome = matmul_gf4(error, stab_list.T)
                recovery, prob_L = decoder.decode(syndrome)
                tmp0 = (error + recovery) % 2
                # num_fail = num_fail + int(code.commute_with_logical(tmp0).max()==1)
                num_fail = num_fail + int(matmul_gf4(tmp0, logical_list.T).max(axis=1).sum())
                num_total = num_total + tmp0.shape[0]
            num_fail_list.append(num_fail)
            num_total_list.append(num_total)
    num_fail_list = np.array(num_fail_list, dtype=np.int64).reshape(len(level_list), -1)
    num_total_list = np.array(num_total_list, dtype=np.int64).reshape(len(level_list), -1)

    fig,ax = plt.subplots()
    for ind0 in range(len(level_list)):
        logical_errors = num_fail_list[ind0]/num_total_list[ind0]
        std_err = (logical_errors*(1-logical_errors)/num_total_list[ind0])**0.5
        ax.errorbar(plist, logical_errors, yerr=std_err, label="level={}".format(level_list[ind0]))
    ax.plot(plist, plist, '-.', label='y=x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    ax.set_title('[[7,1,3]] code, concatenation level')
    ax.legend()
    fig.tight_layout()
    fig.savefig('test.png', dpi=200)

steane_concat_capacity()