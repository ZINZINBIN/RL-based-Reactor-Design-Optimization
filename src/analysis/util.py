import numpy as np
from typing import Dict

def find_feasibile_solutions(result:Dict):

    taus = [comp["tau"] for comp in result["state"]]

    b_limit = result["b_limit"]
    q_limit = result["q_limit"]
    n_limit = result["n_limit"]
    f_limit = result["f_limit"]
    i_limit = result["i_limit"]

    tbr = np.array([result["state"][idx]["TBR"] for idx in range(len(taus))])
    T = np.array([result["control"][idx]["T_avg"] for idx in range(len(taus))])

    taus = np.array(taus)
    b_limit = np.array(b_limit)
    q_limit = np.array(q_limit)
    n_limit = np.array(n_limit)
    f_limit = np.array(f_limit)
    i_limit = np.array(i_limit)

    # Feasible solution
    indices = np.where(((b_limit == 1) * (n_limit == 1) * (q_limit == 1) * (f_limit == 1) * (tbr >= 1)) == 1)

    print("# of feasible solutions with Q > 10:", len(indices[0]))
    print('percentage of feasible solutions: {:.2f}'.format(100 * len(indices[0]) / len(taus)))

    feasible_sols = {}

    for key in result.keys():
        feasible_sols[key] = [result[key][idx] for idx in indices[0]]

    return feasible_sols, indices[0]

def find_optimal_design(result:Dict):

    taus = [comp["tau"] for comp in result["state"]]
    Qs = [comp["Q"] for comp in result["state"]]
    costs = [comp["cost"] for comp in result["state"]]

    b_limit = result["b_limit"]
    q_limit = result["q_limit"]
    n_limit = result["n_limit"]
    f_limit = result["f_limit"]
    i_limit = result["i_limit"]

    tbr = np.array([result["state"][idx]["TBR"] for idx in range(len(taus))])
    T = np.array([result["control"][idx]["T_avg"] for idx in range(len(taus))])

    taus = np.array(taus)
    Qs = np.array(Qs)
    costs = np.array(costs)
    b_limit = np.array(b_limit)
    q_limit = np.array(q_limit)
    n_limit = np.array(n_limit)
    f_limit = np.array(f_limit)
    i_limit = np.array(i_limit)

    # Feasible solution with better performance then reference design
    indices = np.where(((b_limit == 1) * (n_limit == 1) * (q_limit == 1) * (f_limit == 1) * (tbr >= 1) * (Qs >= 10.0) * (taus >= 0.95) * (costs <= 1.0)) == 1)

    if len(indices[0]) < 1:
        return None

    feas_cost = np.array([result["cost"][idx] for idx in indices[0]])
    arg_min = np.argmin(feas_cost)
    arg_min = indices[0][arg_min]

    optimal_result = {}

    for key in result.keys():
        optimal_result[key] = result[key][arg_min]

    for key in result["state"][arg_min].keys():
        optimal_result[key] = result["state"][arg_min][key]

    for key in result["control"][arg_min].keys():
        optimal_result[key] = result["control"][arg_min][key]

    return optimal_result