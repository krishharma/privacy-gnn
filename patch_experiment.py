import re

with open("experiment.py", "r") as f:
    code = f.read()

# 1. Import resource
code = code.replace("from sklearn.metrics import f1_score", "from sklearn.metrics import f1_score\nimport resource")

# 2. Add peak memory, test_f1, actual_dp_epsilon to the final dictionary returned by run_one
# At the end of run_one:
#     "actual_dp_epsilon": _r(dp_epsilon) if defense_name == "dp_sgd" else float("nan"),
# }
old_end = """        "ece_test": _r(ece),
        "actual_dp_epsilon": _r(dp_epsilon) if defense_name == "dp_sgd" else float("nan"),
    }"""
new_end = """        "ece_test": _r(ece),
        "actual_dp_epsilon": _r(dp_epsilon),
        "test_f1": _r(tf),
        "conf_attack_tpr01": _r(ca_tpr01),
        "conf_attack_tpr05": _r(ca_tpr05),
        "threshold_attack_tpr01": _r(ta2_tpr01),
        "threshold_attack_tpr05": _r(ta2_tpr05),
        "shadow_attack_tpr01": _r(sa_tpr01),
        "shadow_attack_tpr05": _r(sa_tpr05),
        "label_only_attack_tpr01": _r(loa_tpr01),
        "label_only_attack_tpr05": _r(loa_tpr05),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2) if hasattr(resource, 'RUSAGE_SELF') else 0.0,
    }"""
code = code.replace(old_end, new_end)

# 3. Initialize the new variables
old_init = """    ca = cac = ta2 = tac2 = float("nan")"""
new_init = """    ca = cac = ca_tpr01 = ca_tpr05 = ta2 = tac2 = ta2_tpr01 = ta2_tpr05 = float("nan")"""
code = code.replace(old_init, new_init)

old_init2 = """    sa = sac = float("nan")
    la = lac = float("nan")
    loa = loac = float("nan")"""
new_init2 = """    sa = sac = sa_tpr01 = sa_tpr05 = float("nan")
    la = lac = float("nan")
    loa = loac = loa_tpr01 = loa_tpr05 = float("nan")"""
code = code.replace(old_init2, new_init2)

old_init3 = """        loa, loac, flip_rate = 0.5, 0.5, 0.0"""
new_init3 = """        loa, loac, loa_tpr01, loa_tpr05, flip_rate = 0.5, 0.5, 0.0, 0.0, 0.0"""
code = code.replace(old_init3, new_init3)

old_init4 = """                sa, sac = 0.5, 0.5"""
new_init4 = """                sa, sac, sa_tpr01, sa_tpr05 = 0.5, 0.5, 0.0, 0.0"""
code = code.replace(old_init4, new_init4)


# 4. Handle Confidence Attack Return
old_conf = """        ca, cac, ta2, tac2 = confidence_attack(
            p[trm], p[tem], yn[trm], yn[tem], random_state=int(seed)
        )"""
new_conf = """        c_res = confidence_attack(
            p[trm], p[tem], yn[trm], yn[tem], random_state=int(seed)
        )
        ca, cac, ca_tpr01, ca_tpr05 = c_res['conf_auc'], c_res['conf_acc'], c_res['conf_tpr_01'], c_res['conf_tpr_05']
        ta2, tac2, ta2_tpr01, ta2_tpr05 = c_res['thresh_auc'], c_res['thresh_acc'], c_res['thresh_tpr_01'], c_res['thresh_tpr_05']"""
code = code.replace(old_conf, new_conf)

# 5. Handle Label-Only Attack Return
old_label = """            loa, loac, dist_m, dist_nm, flip_rate = label_only_attack(target_model, target_data, trm, tem, num_samples=1000, max_noise_scale=5.0)"""
new_label = """            l_res = label_only_attack(target_model, target_data, trm, tem, num_samples=1000, max_noise_scale=5.0)
            loa, loac, loa_tpr01, loa_tpr05, flip_rate = l_res['auc'], l_res['acc'], l_res['tpr_01'], l_res['tpr_05'], l_res['flip_rate']"""
code = code.replace(old_label, new_label)

# 6. Handle Shadow Attack Return
old_shadow = """                sa, sac = shadow_attack(
                    p_sh0[sh_tr0],
                    p_sh0[sh_te0],
                    sh_y[sh_tr0],
                    sh_y[sh_te0],
                    p[trm],
                    p[tem],
                    yn[trm],
                    yn[tem],
                    random_state=int(seed),
                )"""
new_shadow = """                s_res = shadow_attack(
                    p_sh0[sh_tr0], p_sh0[sh_te0], sh_y[sh_tr0], sh_y[sh_te0],
                    p[trm], p[tem], yn[trm], yn[tem], random_state=int(seed),
                )
                sa, sac, sa_tpr01, sa_tpr05 = s_res['auc'], s_res['acc'], s_res['tpr_01'], s_res['tpr_05']"""
code = code.replace(old_shadow, new_shadow)

with open("experiment.py", "w") as f:
    f.write(code)

print("Patched experiment.py")
