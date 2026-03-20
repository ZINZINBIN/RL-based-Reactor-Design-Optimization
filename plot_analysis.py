import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.plot import (
    plot_scatter_feasibility,
    plot_lawson_curve,
    temperal_average
)
from src.optim.util import objective

datapaths = [
    "./results/gridsearch/params_search.pkl",
    "./results/genetic/params_search.pkl",
    "./results/particle/params_search.pkl",
    "./results/bayesian/params_search.pkl",
    "./results/crl/params_search.pkl",
]

savepaths = [
    "./results/gridsearch/",
    "./results/genetic/",
    "./results/particle/",
    "./results/bayesian/",
    "./results/crl/",
]

tags = ["GS", "GA", "PSO", "BO", "DRL"]
clrs = ["r", "g", "m", "b", "k"]

status_list = []

for savepath, datapath, tag, clr in zip(savepaths, datapaths, tags, clrs):

    with open(datapath, "rb") as f:
        result = pickle.load(f)

    tau = np.array([comp["tau"] for comp in result["state"]])
    cost = np.array([comp["cost"] for comp in result["state"]])

    beta = np.array([comp["beta"] for comp in result["state"]])

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
            * (Qs > 10.38)
            * (tau >= 0.95)
            * (cost <= 1.0)
            * (beta <= 5.0)
        )
        == 1
    )

    feasb_ratio = 100 * len(indices[0]) / len(tau)

    if len(indices[0]) == 0:
        print("{}: No feasible solutions found.".format(tag))
        continue

    # The minimum cost case
    feasb_cost = np.array([result["cost"][idx] for idx in indices[0]])

    arg_min = np.argmin(feasb_cost)
    arg_min = indices[0][arg_min]

    # .png file
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

    fig, axes = plot_scatter_feasibility(
        result,
        yparam="Q",
        ylabel=r"$Q$",
        ylims=[8.0, 12.00],
        filename=os.path.join(savepath, "feasibility_Q.png"),
    )

    fig, axes = plot_scatter_feasibility(
        result,
        yparam="cost",
        ylabel="cost",
        ylims=[0.75, 1.25],
        filename=os.path.join(savepath, "feasibility_cost.pdf"),
    )

    fig, axes = plot_scatter_feasibility(
        result,
        yparam="tau",
        ylabel=r"$\tau_E$",
        ylims=[0.75, 1.25],
        filename=os.path.join(savepath, "feasibility_tau.pdf"),
    )

    fig, axes = plot_scatter_feasibility(
        result,
        yparam="Q",
        ylabel=r"$Q$",
        ylims=[8.0, 12.00],
        filename=os.path.join(savepath, "feasibility_Q.pdf"),
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
# .png
plot_lawson_curve(
    filename="./results/compare/lawson_curve.png",
    status_list=status_list,
    xlims=[7.5, 25.0],
    ylims=[0, 8.0],
)

# .pdf
plot_lawson_curve(
    filename="./results/compare/lawson_curve.pdf",
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
    f.write("\nRef: {:.3f}".format(10.380))

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
k = 512

# plot the curve
fig = plt.figure(figsize=(5, 4))
clr = plt.cm.Purples(0.9)

for c, status in zip(clrs, status_list):

    tag = status["tag"]
    reward = [objective(state, discretized = False) for state in status["state"]]
    episode = np.array(range(1, len(reward) + 1, 1))

    reward = np.clip(reward, a_min=y_min, a_max=y_max)
    reward_mean, reward_lower, reward_upper = temperal_average(reward, k)

    plt.plot(episode, reward_mean, c=c, label=tag)

plt.xlabel("Iteration")
plt.ylabel(r"$<R_{\text{total}}>$")
plt.legend(loc="upper left")
fig.tight_layout()

# .png file
plt.savefig("./results/compare/optimization_process.png", dpi=120)

# .pdf file
plt.savefig("./results/compare/optimization_process.pdf", dpi=120)
plt.close()

# Feasibility as evaluation
fig = plt.figure(figsize=(5, 4))
clr = plt.cm.Purples(0.9)

for c, status in zip(clrs, status_list):

    tag = status["tag"]
    feasb_score = [objective(state, [0,0,0], discretized = False) for state in status["state"]]
    episode = np.array(range(1, len(reward) + 1, 1))

    feasb_score, _, _ = temperal_average(feasb_score, k)
    plt.plot(episode, feasb_score, c=c, label=tag)

plt.xlabel("Iteration")
plt.ylabel(r"$<-\lambda \cdot C(s_t)>$")
plt.legend(loc="upper left")
fig.tight_layout()

# .png file
plt.savefig("./results/compare/optimization_process_feasibility.png", dpi=120)

# .pdf file
plt.savefig("./results/compare/optimization_process_feasibility.pdf", dpi=120)
plt.close()

# plot the partial reward curve (tau, cost, Q)
fig, axes = plt.subplots(1, 3, figsize = (15, 4), sharex=True)
axes = axes.ravel()
clr = plt.cm.Purples(0.9)

for c, status in zip(clrs, status_list):

    tag = status["tag"]

    reward_tau = [objective(state, [1,0,0], [0,0,0,0,0]) for state in status["state"]]
    reward_cost = [objective(state, [0,1,0], [0,0,0,0,0]) for state in status["state"]]
    reward_Q = [objective(state, [0,0,1], [0,0,0,0,0]) for state in status["state"]]

    episode = np.array(range(1, len(reward) + 1, 1))

    reward_tau = np.clip(reward_tau, a_min=y_min, a_max=y_max)
    reward_cost = np.clip(reward_cost, a_min=y_min, a_max=y_max)
    reward_Q = np.clip(reward_Q, a_min=y_min, a_max=y_max)

    reward_tau, _, _ = temperal_average(reward_tau, k)
    reward_cost, _, _ = temperal_average(reward_cost, k)
    reward_Q, _, _ = temperal_average(reward_Q, k)

    axes[0].plot(episode, reward_tau, c=c, label=tag)
    axes[1].plot(episode, reward_cost, c=c, label=tag)
    axes[2].plot(episode, reward_Q, c=c, label=tag)

axes[0].set_xlabel("Iteration")
axes[1].set_xlabel("Iteration")
axes[2].set_xlabel("Iteration")

axes[0].set_ylabel(r"$<R_{tau}>$")
axes[1].set_ylabel(r"$<R_{cost}>$")
axes[2].set_ylabel(r"$<R_{Q}>$")

axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left")
axes[2].legend(loc="upper left")
fig.tight_layout()

# .png file
plt.savefig("./results/compare/optimization_process_partial.png", dpi=120)

# .pdf file
plt.savefig("./results/compare/optimization_process_partial.pdf", dpi=120)
plt.close()

# plot the partial penalty terms
fig, axes = plt.subplots(2, 2, figsize = (10, 8), sharex=True)
axes = axes.ravel()
clr = plt.cm.Purples(0.9)

for c, status in zip(clrs, status_list):

    tag = status["tag"]

    # beta / Kink / Greenwald / Bootstrap
    reward_b = [objective(state, [0,0,0], [1,0,0,0,0]) for state in status["state"]]
    reward_q = [objective(state, [0,0,0], [0,1,0,0,0]) for state in status["state"]]
    reward_n = [objective(state, [0,0,0], [0,0,1,0,0]) for state in status["state"]]
    reward_f = [objective(state, [0,0,0], [0,0,0,1,0]) for state in status["state"]]

    episode = np.array(range(1, len(reward) + 1, 1))

    reward_b = np.clip(reward_b, a_min=y_min, a_max=y_max)
    reward_q = np.clip(reward_q, a_min=y_min, a_max=y_max)
    reward_n = np.clip(reward_n, a_min=y_min, a_max=y_max)
    reward_f = np.clip(reward_f, a_min=y_min, a_max=y_max)

    reward_b, _, _ = temperal_average(reward_b, k)
    reward_q, _, _ = temperal_average(reward_q, k)
    reward_n, _, _ = temperal_average(reward_n, k)
    reward_f, _, _ = temperal_average(reward_f, k)

    axes[0].plot(episode, reward_b, c=c, label=tag)
    axes[1].plot(episode, reward_q, c=c, label=tag)
    axes[2].plot(episode, reward_n, c=c, label=tag)
    axes[3].plot(episode, reward_f, c=c, label=tag)

axes[0].set_xlabel("Iteration")
axes[1].set_xlabel("Iteration")
axes[2].set_xlabel("Iteration")
axes[3].set_xlabel("Iteration")

axes[0].set_ylabel(r"$-C_{\beta}$")
axes[1].set_ylabel(r"$-C_{q}$")
axes[2].set_ylabel(r"$-C_{G}$")
axes[3].set_ylabel(r"$-C_{\text{bs}}$")

axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left")
axes[2].legend(loc="upper left")
axes[3].legend(loc="upper left")
fig.tight_layout()

# .png file
plt.savefig("./results/compare/optimization_process_feasibility_partial.png", dpi=120)

# .pdf file
plt.savefig("./results/compare/optimization_process_feasibility_partial.pdf", dpi=120)
plt.close()
