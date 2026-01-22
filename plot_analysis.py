import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.plot import (
    plot_scatter_feasibility,
    plot_lawson_curve,
    temperal_average,
)
from src.optim.util import objective

datapaths = [
    "./results/gridsearch/params_search.pkl",
    "./results/genetic/params_search.pkl",
    "./results/particle/params_search.pkl",
    "./results/bayesian/params_search.pkl",
    "./results/rl/params_search.pkl",
    "./results/crl/params_search.pkl",
]

savepaths = [
    "./results/gridsearch/",
    "./results/genetic/",
    "./results/particle/",
    "./results/bayesian/",
    "./results/rl/",
    "./results/crl/",
]

tags = ["Gridsearch", "Genetic", "Particle", "Bayesian", "RL", "CRL"]

clrs = ["r", "g", "m", "b", "k", "c"]

status_list = []

for savepath, datapath, tag, clr in zip(savepaths, datapaths, tags, clrs):

    with open(datapath, "rb") as f:
        result = pickle.load(f)

    tau = np.array([comp["tau"] for comp in result["state"]])
    cost = np.array([comp["cost"] for comp in result["state"]])
    
    b_limit = result["b_limit"]
    q_limit = result["q_limit"]
    n_limit = result["n_limit"]
    f_limit = result["f_limit"]
    i_limit = result["i_limit"]
    tbr = np.array([result["state"][idx]["TBR"] for idx in range(len(tau))])
    T = np.array([result["control"][idx]["T_avg"] for idx in range(len(tau))])

    tau = np.array(tau)
    b_limit = np.array(b_limit)
    q_limit = np.array(q_limit)
    n_limit = np.array(n_limit)
    f_limit = np.array(f_limit)
    i_limit = np.array(i_limit)

    Qs = np.array([comp["Q"] for comp in result["state"]])
    indices = np.where(
        (
            (b_limit == 1)
            * (n_limit == 1)
            * (q_limit == 1)
            * (f_limit == 1)
            * (tbr >= 1)
            * (Qs > 10.0)
            * (tau >= 0.95)
            * (cost <= 1.0)
        )
        == 1
    )

    feasb_ratio = 100 * len(indices[0]) / len(tau)

    if len(indices[0]) == 0:
        print("No feasible solutions found.")
        continue

    # The minimum cost case
    feasb_cost = np.array([result["cost"][idx] for idx in indices[0]])

    arg_min = np.argmin(feasb_cost)
    arg_min = indices[0][arg_min]

    fig, axes = plot_scatter_feasibility(
        result,
        yparam="cost",
        ylabel="cost",
        ylims=[0.75, 1.25],
        filename=os.path.join(savepath, "feasibility_cost.png"),
    )

    fig, axes = plot_scatter_feasibility(
        result,
        yparam="tau",
        ylabel=r"$\tau_E$",
        ylims=[0.75, 1.25],
        filename=os.path.join(savepath, "feasibility_tau.png"),
    )

    # save status
    n_operation = result["state"][arg_min]["n"]
    tau_operation = result["state"][arg_min]["tau"]
    T_operation = result["control"][arg_min]["T_avg"]
    Q_operation = result["state"][arg_min]["Q"]
    cost_operation = result["state"][arg_min]["cost"]

    status = {
        "n": n_operation,
        "T": T_operation,
        "tau": tau_operation,
        "Q": Q_operation,
        "tag": tag,
        "c": clr,
        "cost": cost_operation,
        "state": result["state"],
        "feasbility_ratio": feasb_ratio,
    }

    status_list.append(status)


# Lawson curve
plot_lawson_curve(
    filename="./results/compare/lawson_curve.png",
    status_list=status_list,
    xlims=[7.5, 25.0],
    ylims=[0, 8.0],
)

# Design info
with open("./results/compare/design_spec.txt", "w") as f:
    f.write("\n================================================")
    f.write("\n============= Optimization result ==============")
    f.write("\n================================================\n")
    f.write("\n==================== Cost ======================\n")
    f.write("\nRef: {:.3f}".format(1.003))

    for status in status_list:
        tag = status["tag"]
        f.write("\n{}: {:.3f}".format(tag, status["cost"]))

    f.write("\n")
    f.write("\n================ Confinement ====================\n")
    f.write("\nRef: {:.3f}".format(0.944))

    for status in status_list:
        tag = status["tag"]
        f.write("\n{}: {:.3f}".format(tag, status["tau"]))

    f.write("\n")
    f.write("\n================ Energy gain Q ==================\n")
    f.write("\nRef: {:.3f}".format(10.0))

    for status in status_list:
        tag = status["tag"]
        f.write("\n{}: {:.3f}".format(tag, status["Q"]))

    f.write("\n")
    f.write("\n======== Feasible solutions percentage ==========\n")

    for status in status_list:
        tag = status["tag"]
        f.write("\n{}: {:.2f}".format(tag, status["feasbility_ratio"]))

# Time evolving optimization iteration process
# clip the invalid value
y_min = 0.0
y_max = 5.0

# temperal average
k = 128

# plot the curve
fig = plt.figure(figsize=(6, 5))
clr = plt.cm.Purples(0.9)

for c, status in zip(clrs, status_list):

    tag = status["tag"]
    reward = [objective(state) for state in status["state"]]
    episode = np.array(range(1, len(reward) + 1, 1))

    reward = np.clip(reward, a_min=y_min, a_max=y_max)
    reward_mean, reward_lower, reward_upper = temperal_average(reward, k)

    plt.plot(episode, reward_mean, c=c, label=tag)

plt.xlabel("Episodes")
plt.ylabel(r"$\bar{R}(s_t)$")
plt.ylim([0.9, 1.75])
plt.legend(loc="upper right")
fig.tight_layout()
plt.savefig("./results/compare/optimization_process.png", dpi=120)
