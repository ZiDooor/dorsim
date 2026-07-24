import numpy as np
from tqdm import tqdm
from dorsim import (
    Circuit,
    PauliFrame,
    BiasedPoulinDecoder,
    StabilizerCode,
    concat_code,
)
import matplotlib.pyplot as plt


px_list = [[0.03842147435897436, 0.05151241987179487, 0.06683693910256411, 0.07679286858974359, 0.0894931891025641, 0.10228365384615384], [0.04243790064102564, 0.05601963141025641, 0.06958133012820512, 0.0859375, 0.09841746794871795, 0.11135817307692308], [0.04273838141025641, 0.05586939102564103, 0.07034254807692308, 0.08471554487179488, 0.10248397435897436, 0.11274038461538462]]
py_list = [[0.014723557692307692, 0.021694711538461538, 0.02736378205128205, 0.03728966346153846, 0.04276842948717949, 0.04830729166666667], [0.01461338141025641, 0.02201522435897436, 0.032782451923076925, 0.04178685897435897, 0.05388621794871795, 0.05904447115384615], [0.015094150641025641, 0.021604567307692307, 0.03177083333333333, 0.0401943108974359, 0.053335336538461536, 0.06095753205128205]]
pz_list = [[0.02548076923076923, 0.040184294871794875, 0.05450721153846154, 0.08196113782051281, 0.09409054487179487, 0.10266426282051282], [0.01867988782051282, 0.03748998397435897, 0.06403245192307692, 0.08296274038461539, 0.10114182692307692, 0.1155048076923077], [0.017157451923076925, 0.035556891025641024, 0.06272035256410256, 0.08704927884615385, 0.10384615384615385, 0.11709735576923076]]


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
    plist = np.linspace(0.02, 0.06, 6)
    batch_size = 512


    num_fail_list = []
    num_total_list = []
    num_fail_list_biased = []
    num_total_list_biased = []

    for level_i in level_list:
        code = code_list[level_i-1]
        stab_list = code.stabilizers
        logical_list = np.concatenate([code.logical_x, code.logical_z], axis=0)

        for ind_p, p in enumerate(plist):
            num_fail = 0
            num_total = 0
            num_fail_biased = 0
            num_total_biased = 0

            for _ in tqdm(range(num_sample//batch_size), desc=f"level={level_i}, p={p:.3f}"):
                re = one_task(level_i, ind_p, code, p, batch_size, stab_list, logical_list)
                num_fail += re[0][0]
                num_total += re[0][1]
                num_fail_biased += re[1][0]
                num_total_biased += re[1][1]

            num_fail_list.append(num_fail)
            num_total_list.append(num_total)
            num_fail_list_biased.append(num_fail_biased)
            num_total_list_biased.append(num_total_biased)
    num_fail_list = np.array(num_fail_list, dtype=np.int64).reshape(len(level_list), -1)
    num_total_list = np.array(num_total_list, dtype=np.int64).reshape(len(level_list), -1)
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

def one_task(level, ind_p, code, p, batch, stab, logical):
    ind_tmp = level - 1
    
    p_a = p
    p_b = p*2
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

    decoder_tele = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)
    decoder_end = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)

    #### Unbiased decoding
    pframe = PauliFrame(circuit=circ, shots=batch)
    pframe.frame.fill(0) # turn it into code capacity simulation
    pframe.run()
    error = pframe.samples
    pframe.select_qubits(ind_q[2*n_q:]) # select all qubits
    frame = pframe.frame
    ## Knill correction
    syn0 = matmul_gf4(error, stab.T)
    decoder_tele.set_error_model(p/3, p/3, p/3) ########### set the ECT error distribution
    re0, prob_L = decoder_tele.decode(syn0)
    tmp0 = (frame + (error + re0)) % 2
    ## Count the failures
    syn1 = matmul_gf4(tmp0, stab.T)
    decoder_end.set_error_model(p/3, p/3, p/3) ########### set the output state distribution
    re1, prob_L = decoder_end.decode(syn1)
    tmp1 = (tmp0 + re1) % 2

    check_list = np.concatenate([stab, logical], axis=0)
    num_fail = int(matmul_gf4(tmp1, check_list.T).max(axis=1).sum())
    num_total = tmp0.shape[0]

    #### Biased decoding
    pframe = PauliFrame(circuit=circ, shots=batch)
    pframe.frame.fill(0) # turn it into code capacity simulation
    pframe.run()
    error = pframe.samples
    pframe.select_qubits(ind_q[2*n_q:]) # select all qubits
    frame = pframe.frame
    ## Knill correction
    syn0 = matmul_gf4(error, stab.T)
    ########### set the ECT error distribution
    decoder_tele.set_error_model((1 - r_a*r_b)/4, (1 - r_a*r_b)/4, (1 + r_a*r_b - 2*r_a*r_b**2)/4)
    re0, prob_L = decoder_tele.decode(syn0)
    tmp0 = (frame + (error + re0)) % 2
    ## Count the failures
    syn1 = matmul_gf4(tmp0, stab.T)
    ########### set the output state distribution
    # p_cor = [-2*p**2, 6*p**2, (149/2)*p**2]
    p_cor = [0, 0, 0]
    decoder_end.set_error_model((1 + r_b - 2*r_b**2)/4 + p_cor[0], (1 - r_b)/4 + p_cor[1], (1 - r_b)/4 + p_cor[2])
    # decoder_end.set_error_model(px_list[ind_tmp][ind_p], py_list[ind_tmp][ind_p], pz_list[ind_tmp][ind_p])
    re1, prob_L = decoder_end.decode(syn1)
    tmp1 = (tmp0 + re1) % 2

    check_list = np.concatenate([stab, logical], axis=0)
    num_fail_biased = int(matmul_gf4(tmp1, check_list.T).max(axis=1).sum())
    num_total_biased = tmp0.shape[0]

    return [[num_fail, num_total],
            [num_fail_biased, num_total_biased]]


steane_concat_capacity()