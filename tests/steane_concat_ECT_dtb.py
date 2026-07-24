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

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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


    num_fail_x_list = []
    num_fail_y_list = []
    num_fail_z_list = []
    num_total_list = []

    for level_i in level_list:
        code = code_list[level_i-1]
        stab_list = code.stabilizers
        logical_list = np.concatenate([code.logical_x, code.logical_z], axis=0)

        decoder_tele = BiasedPoulinDecoder(code, 1/4, 1/4, 1/4)
        for p in plist:
            num_fail_x = 0
            num_fail_y = 0
            num_fail_z = 0
            num_total = 0

            for _ in tqdm(range(num_sample//batch_size), desc=f"level={level_i}, p={p:.3f}"):
                re = one_task(code, p, batch_size, decoder_tele, stab_list) # return num_fail for X, Y, Z and num_total

                num_fail_x += re[0]
                num_fail_y += re[1]
                num_fail_z += re[2]
                num_total += re[3]

            num_fail_x_list.append(num_fail_x)
            num_fail_y_list.append(num_fail_y)
            num_fail_z_list.append(num_fail_z)
            num_total_list.append(num_total)
    num_fail_x_list = np.array(num_fail_x_list, dtype=np.int64).reshape(len(level_list), -1)
    num_fail_y_list = np.array(num_fail_y_list, dtype=np.int64).reshape(len(level_list), -1)
    num_fail_z_list = np.array(num_fail_z_list, dtype=np.int64).reshape(len(level_list), -1)
    num_total_list = np.array(num_total_list, dtype=np.int64).reshape(len(level_list), -1)

    print(f"px_list = {(num_fail_x_list/num_total_list).tolist()}")
    print(f"py_list = {(num_fail_y_list/num_total_list).tolist()}")
    print(f"pz_list = {(num_fail_z_list/num_total_list).tolist()}")

    # fig,ax = plt.subplots()
    # for ind0 in range(len(level_list)):
    #     color = color_list[ind0]
    #     x_errors = num_fail_x_list[ind0]/num_total_list[ind0]
    #     std_err_x = (x_errors*(1-x_errors)/num_total_list[ind0])**0.5
    #     y_errors = num_fail_y_list[ind0]/num_total_list[ind0]
    #     std_err_y = (y_errors*(1-y_errors)/num_total_list[ind0])**0.5
    #     z_errors = num_fail_z_list[ind0]/num_total_list[ind0]
    #     std_err_z = (z_errors*(1-z_errors)/num_total_list[ind0])**0.5
    #     ax.errorbar(plist, x_errors, yerr=std_err_x, ecolor=color, label="level={}, X".format(level_list[ind0]))
    #     ax.errorbar(plist, y_errors, yerr=std_err_y, ecolor=color, linestyle='--', label="level={}, Y".format(level_list[ind0]))
    #     ax.errorbar(plist, z_errors, yerr=std_err_z, ecolor=color, linestyle='-.', label="level={}, Z".format(level_list[ind0]))
    # # ax.plot(plist, plist, '-.', label='y=x')
    # # ax.set_xlim(0, 5)
    # # ax.set_ylim(0.1, 0.7)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.grid()
    # ax.set_title('[[7,1,3]] code, concatenation level')
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig('test.png', dpi=200)

def one_task(code, p, batch, decoder_tele, stab):
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
    frame_out = (frame + (error + re0)) % 2
    error_out = frame_out[:, :n_q] + frame_out[:, n_q:]*2 # I: 0, X: 1, Z: 2, Y: 3
    ## Count the failures
    ind_q = 0 # check the first physical qubit
    num_x = (error_out[:, ind_q] == 1).sum()
    num_y = (error_out[:, ind_q] == 3).sum()
    num_z = (error_out[:, ind_q] == 2).sum()
    num = error_out.shape[0]

    return (num_x, num_y, num_z, num)


steane_concat_capacity()