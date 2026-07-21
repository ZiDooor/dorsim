import itertools
import numpy as np
from tqdm import tqdm
from dorsim import (
    Circuit,
    PauliFrame,
    BiasedPoulinDecoder,
    CSSCode,
    StabilizerCode,
    concat_code,
)
import matplotlib.pyplot as plt

def matmul_gf4(np0, np1):
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
    plist = np.linspace(0.06, 0.10, 6)
    batch_size = 512


    ### Unbiased decoding
    num_fail_list = []
    num_total_list = []
    for level_i in level_list:
        code = code_list[level_i-1]
        stab_list = code.stabilizers
        logical_list = np.concatenate([code.logical_x, code.logical_z], axis=0)

        decoder_tele = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)
        decoder_end = decoder_tele
        for p in plist:
            num_fail = 0
            num_total = 0

            p_a = p
            p_b = p
            r_a = 1 - 4*p_a/3
            r_b = 1 - 4*p_b/3

            n_q = code.n
            
            ind_q = np.arange(3*n_q)
            circ = (
                Circuit(3*n_q)
                .depolarize1(ind_q[:n_q], p_a) # depolarizing channel, input state
                .depolarize1(ind_q[n_q:], p_b) # depolarizing channel, Bell pair, maybe 1% smaller?
                .cx(np.column_stack((ind_q[n_q:2*n_q], ind_q[2*n_q:])).ravel())
                .cx(np.column_stack((ind_q[:n_q], ind_q[n_q:2*n_q])).ravel())
                .h(ind_q[:n_q])
                .m(ind_q[n_q:2*n_q]) # measure X errors
                .m(ind_q[:n_q]) # measure Z errors
            )

            for _ in tqdm(range(num_sample//batch_size), desc=f'level {level_i} p={p:.3f}'):

                pframe = PauliFrame(circuit=circ, shots=batch_size)
                pframe.frame.fill(0) # turn it into code capacity simulation
                pframe.run()
                error = pframe.samples
                pframe.select_qubits(ind_q[2*n_q:]) # select all qubits
                frame = pframe.frame
                ## Knill correction
                syn0 = matmul_gf4(error, stab_list.T)
                decoder_tele.set_error_model(p/3, p/3, p/3) ########### set the ECT error distribution
                re0, prob_L = decoder_tele.decode(syn0)
                tmp0 = (frame + (error + re0)) % 2
                ## Count the failures
                syn1 = matmul_gf4(tmp0, stab_list.T)
                # decoder_end.set_error_model(p/3, p/3, p/3) ########### set the output state distribution
                re1, prob_L = decoder_end.decode(syn1)
                tmp1 = (tmp0 + re1) % 2

                check_list = np.concatenate([stab_list, logical_list], axis=0)
                num_fail = num_fail + int(matmul_gf4(tmp1, check_list.T).max(axis=1).sum())
                num_total = num_total + tmp0.shape[0]
            num_fail_list.append(num_fail)
            num_total_list.append(num_total)
    num_fail_list = np.array(num_fail_list, dtype=np.int64).reshape(len(level_list), -1)
    num_total_list = np.array(num_total_list, dtype=np.int64).reshape(len(level_list), -1)


    ### Biased decoding
    num_fail_list_biased = []
    num_total_list_biased = []
    for level_i in level_list:
        code = code_list[level_i-1]
        stab_list = code.stabilizers
        logical_list = np.concatenate([code.logical_x, code.logical_z], axis=0)

        decoder_tele = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)
        decoder_end = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)
        for p in plist:
            num_fail = 0
            num_total = 0

            p_a = p
            p_b = p
            r_a = 1 - 4*p_a/3
            r_b = 1 - 4*p_b/3

            n_q = code.n
            
            ind_q = np.arange(3*n_q)
            circ = (
                Circuit(3*n_q)
                .depolarize1(ind_q[:n_q], p_a) # depolarizing channel, input state
                .depolarize1(ind_q[n_q:], p_b) # depolarizing channel, Bell pair, maybe 1% smaller?
                .cx(np.column_stack((ind_q[n_q:2*n_q], ind_q[2*n_q:])).ravel())
                .cx(np.column_stack((ind_q[:n_q], ind_q[n_q:2*n_q])).ravel())
                .h(ind_q[:n_q])
                .m(ind_q[n_q:2*n_q]) # measure X errors
                .m(ind_q[:n_q]) # measure Z errors
            )

            for _ in tqdm(range(num_sample//batch_size), desc=f'level {level_i} p={p:.3f}'):

                pframe = PauliFrame(circuit=circ, shots=batch_size)
                pframe.frame.fill(0) # turn it into code capacity simulation
                pframe.run()
                error = pframe.samples
                pframe.select_qubits(ind_q[2*n_q:]) # select all qubits
                frame = pframe.frame
                ## Knill correction
                syn0 = matmul_gf4(error, stab_list.T)
                ########### set the ECT error distribution
                decoder_tele.set_error_model((1 - r_a*r_b)/4, (1 - r_a*r_b)/4, (1 + r_a*r_b - 2*r_a*r_b**2)/4)
                re0, prob_L = decoder_tele.decode(syn0)
                tmp0 = (frame + (error + re0)) % 2
                ## Count the failures
                syn1 = matmul_gf4(tmp0, stab_list.T)
                ########### set the output state distribution
                decoder_end.set_error_model((1 - r_a*r_b**2)/4, (1 - 2*r_a*r_b + r_a*r_b**2)/4, (1 - r_a*r_b**2)/4)
                re1, prob_L = decoder_end.decode(syn1)
                tmp1 = (tmp0 + re1) % 2

                check_list = np.concatenate([stab_list, logical_list], axis=0)
                num_fail = num_fail + int(matmul_gf4(tmp1, check_list.T).max(axis=1).sum())
                num_total = num_total + tmp0.shape[0]
            num_fail_list_biased.append(num_fail)
            num_total_list_biased.append(num_total)
    num_fail_list_biased = np.array(num_fail_list_biased, dtype=np.int64).reshape(len(level_list), -1)
    num_total_list_biased = np.array(num_total_list_biased, dtype=np.int64).reshape(len(level_list), -1)


    fig,ax = plt.subplots()
    for ind0 in range(len(level_list)):
        logical_errors = num_fail_list[ind0]/num_total_list[ind0]
        std_err = (logical_errors*(1-logical_errors)/num_total_list[ind0])**0.5
        ax.errorbar(plist, logical_errors, yerr=std_err, label="level={}".format(level_list[ind0]))
    for ind0 in range(len(level_list)):
        logical_errors = num_fail_list_biased[ind0]/num_total_list_biased[ind0]
        std_err = (logical_errors*(1-logical_errors)/num_total_list_biased[ind0])**0.5
        ax.errorbar(plist, logical_errors, yerr=std_err, linestyle='--', label="level={}".format(level_list[ind0]))
    ax.plot(plist, plist, '-.', label='y=x')
    # ax.set_xlim(0, 5)
    # ax.set_ylim(0.1, 0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    ax.set_title('[[7,1,3]] code, concatenation level')
    ax.legend()
    fig.tight_layout()
    fig.savefig('test.png', dpi=200)

steane_concat_capacity()