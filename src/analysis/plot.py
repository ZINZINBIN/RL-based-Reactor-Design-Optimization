import numpy as np
import os
import matplotlib.pyplot as plt
from src.design.lawson import Lawson
from src.config.search_space_info import search_space, state_space
from typing import Optional, List, Dict, Callable
from scipy.signal import savgol_filter

ctrl_param_name = list(search_space.keys())
state_param_name = list(state_space.keys())

def temperal_average(Y: np.array, k: int):

    maxlen = len(Y)
    Y_mean = np.zeros(maxlen)
    Y_lower = np.zeros(maxlen)
    Y_upper = np.zeros(maxlen)

    for i in range(maxlen):

        idx_start = i - k // 2
        idx_end = i + k // 2

        if idx_end >= maxlen:
            idx_end = maxlen - 1

        if idx_start <0:
            idx_start = 0

        Y_mean[i] = np.mean(Y[idx_start:idx_end])
        Y_lower[i] = np.min(Y[idx_start:idx_end])
        Y_upper[i] = np.max(Y[idx_start:idx_end])

    return Y_mean, Y_lower, Y_upper


def hampel_filter(y, window_size=15, n_sigmas=3.0, return_mask=False):
    """
    Hampel filter for 1D time series.
    Flags points far from rolling median by n_sigmas * MAD, and replaces them with the median.

    Parameters
    ----------
    y : array-like, shape (N,)
        Time series values.
    window_size : int
        Half-window size on each side (total window = 2*window_size+1).
    n_sigmas : float
        Outlier threshold in robust sigmas.
    return_mask : bool
        If True, also return a boolean mask of detected outliers.

    Returns
    -------
    y_filt : np.ndarray
        Filtered series with outliers replaced.
    outlier_mask : np.ndarray (optional)
        True where an outlier was detected.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    y_filt = y.copy()
    outlier_mask = np.zeros(n, dtype=bool)

    # Scale factor: MAD -> robust std estimate for normal distribution
    k = 1.4826

    for i in range(n):
        i0 = max(0, i - window_size)
        i1 = min(n, i + window_size + 1)
        window = y[i0:i1]

        med = np.nanmedian(window)
        mad = np.nanmedian(np.abs(window - med))

        if mad == 0 or not np.isfinite(mad):
            continue

        robust_sigma = k * mad
        if np.abs(y[i] - med) > n_sigmas * robust_sigma:
            outlier_mask[i] = True
            y_filt[i] = med

    return (y_filt, outlier_mask) if return_mask else y_filt

def remove_peak_targeted(t, y, *,
                         hampel_half_window=30,
                         n_sigmas=4.0,
                         pad=20,
                         smooth_window=81,
                         smooth_poly=3):
    """
    Remove a localized peak from a single time series y(t) while keeping the rest intact.

    Steps:
      1) Light smooth -> baseline
      2) Detect outliers in residual via Hampel-like rule (rolling median & MAD)
      3) Expand detected region by 'pad' samples (captures full peak shoulders)
      4) Replace that region by linear interpolation between boundary points
      5) Optionally apply mild Savitzky-Golay to remove tiny kinks (blue only)

    Parameters are in samples (not time units).
    """

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    # --- (A) baseline for residual (light smoothing) ---
    # Ensure odd window <= n
    sw = min(smooth_window, n - (1 - n % 2))
    if sw % 2 == 0:
        sw -= 1
    baseline = savgol_filter(y, window_length=sw, polyorder=min(smooth_poly, sw - 1))

    resid = y - baseline

    # --- (B) Hampel-style detection on residual ---
    k = 1.4826  # MAD->sigma factor
    outlier = np.zeros(n, dtype=bool)

    for i in range(n):
        i0 = max(0, i - hampel_half_window)
        i1 = min(n, i + hampel_half_window + 1)
        w = resid[i0:i1]

        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad <= 1e-14:
            continue

        sigma = k * mad
        if np.abs(resid[i] - med) > n_sigmas * sigma:
            outlier[i] = True

    if not outlier.any():
        # nothing detected; return lightly smoothed version (or original if you prefer)
        return y.copy(), outlier

    # --- (C) Expand region (so you remove the whole peak, not just a few points) ---
    idx = np.where(outlier)[0]
    left = max(0, idx[0] - pad)
    right = min(n - 1, idx[-1] + pad)

    y_fixed = y.copy()

    # --- (D) Replace [left:right] by interpolation between boundaries ---
    # pick boundary anchors just outside the replaced region
    L = max(0, left - 1)
    R = min(n - 1, right + 1)

    if R <= L + 1:
        return y.copy(), outlier  # degenerate case

    # Fill the region using linear interpolation on the original y
    fill_idx = np.arange(L, R + 1)
    y_fixed[L:R + 1] = np.interp(fill_idx, [L, R], [y[L], y[R]])

    # --- (E) Mild smoothing after fixing (keeps the overall shape) ---
    sw2 = min(smooth_window, n - (1 - n % 2))
    if sw2 % 2 == 0:
        sw2 -= 1
    y_final = savgol_filter(y_fixed, window_length=sw2, polyorder=min(smooth_poly, sw2 - 1))

    # Important: keep the unmodified parts as close as possible to original.
    # If you want *zero* smoothing outside the peak, comment out y_final above and return y_fixed.
    return y_final, (left, right)

def plot_loss_curve(
    loss_list:List, 
    buffer_size : int, 
    k:int = 8, 
    y_min:float = -2.0,
    y_max:float = 5.0,
    save_dir : Optional[str] = None,
    ):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    loss = np.repeat(np.array(loss_list).reshape(-1,1), repeats = buffer_size, axis = 1).reshape(-1,)
    episode = np.array(range(1, len(loss)+1, 1))
    
    # clip the invalid value
    loss = np.clip(loss, a_min = y_min, a_max = y_max)
    
    # temperal average
    loss_mean, loss_lower, loss_upper = temperal_average(loss, k)
    
    fig = plt.figure(figsize = (6,4))
    
    clr = plt.cm.Purples(0.9)
    
    plt.plot(episode, loss_mean, c = 'r', label = r'$<L_\theta^{\pi}(t)>$')
    plt.fill_between(episode, loss_lower, loss_upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
    
    plt.xlabel("Episodes")
    plt.ylabel(r"Policy Loss $L_\theta^{\pi}$")
    plt.legend(loc = 'upper right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_loss.png"), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    fig.clear()

def plot_reward(
    state_list:List,
    objective:Callable,
    k:int = 8, 
    y_min:float = -1.0,
    y_max:float = 10.0,
    save_dir : Optional[str] = None,
):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    reward = [objective(state) for state in state_list]
    episode = np.array(range(1, len(reward)+1, 1))
    
    # clip the invalid value
    reward = np.clip(reward, a_min = y_min, a_max = y_max)
    
    # temperal average
    reward_mean, reward_lower, reward_upper = temperal_average(reward, k)
    
    fig = plt.figure(figsize = (6,4))
    
    clr = plt.cm.Purples(0.9)
    
    plt.plot(episode, reward_mean, c = 'r', label = r'$<R(s,a)_t>$')
    plt.fill_between(episode, reward_lower, reward_upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
    
    plt.xlabel("Episodes")
    plt.ylabel(r"Reward $R(s,a)$")
    plt.legend(loc = 'upper right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward.png"), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    fig.clear()

def plot_lawson_curve(
    filename:Optional[str],
    status_list:Optional[List],
    xlims:List = [5,100],
    ylims:List = [0,10]
):

    lawson = Lawson()

    # Reference case
    Q_reference = 10.38
    n_reference = 1.43
    tau_reference = 0.944
    T_reference = 14.0
    B_reference = 13.0
    
    R = 5.346
    a = 1.337
    b = 1.199
    
    # Lawson curve with reference setup
    T = np.linspace(6, 100, 64, endpoint=False)
    n_rescale = n_reference * 10**20
    B_rescale = B_reference * (1 - (a+b)/R)

    psi = 10 ** (-3)

    Q = 10

    n_tau = [lawson.compute_n_tau_lower_bound(t, n_rescale, B_rescale, psi) * 10 ** (-20) for t in T]
    n_tau_5 = [lawson.compute_n_tau_Q_lower_bound(t, n_rescale, B_rescale, psi, 5) * 10 ** (-20) for t in T]
    n_tau_Q = [lawson.compute_n_tau_Q_lower_bound(t, n_rescale, B_rescale, psi, Q) * 10 ** (-20) for t in T]
    n_tau_break = [lawson.compute_n_tau_Q_lower_bound(t, n_rescale, B_rescale, psi, 1) * 10 ** (-20) for t in T]

    fig, ax = plt.subplots(1,1, figsize = (6,5))
    ax.plot(T, n_tau, "k", label = "Lawson criteria (Ignition)")
    ax.plot(T, n_tau_5, "r", label = "Lawson criteria (Q=5)")
    ax.plot(T, n_tau_Q, "b", label = "Lawson criteria (Q={})".format(Q))
    ax.plot(T, n_tau_break, "g", label = "Lawson criteria (Breakeven)")
    
    ax.scatter(T_reference, tau_reference * n_reference, c = 'k', label = 'Reference (Q={:.2f})'.format(Q_reference))

    if status_list is not None:
        
        for status in status_list:
            
            tag = status['tag']
            design_T = status['T']
            design_tau = status['tau']
            design_n = status['n']
            design_Q = status['Q']
            clr = status['c']
            
            ax.scatter(design_T, design_tau * design_n, c = clr, label = '{} (Q={:.2f})'.format(tag, design_Q))

    ax.set_xlabel("T(unit : keV)")
    ax.set_ylabel("$(N\\tau_E)_{dt}(unit:10^{20}s * m^{-3})$")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(loc = "upper right")
    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    
    plt.close()

    return fig, ax


def plot_scatter(
    result:Dict,
    xparam: str,
    yparam: str,
    xlims: List,
    ylims: List,
    filename: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):

    if xparam in ctrl_param_name:
        xs = [comp[xparam] for comp in result['control']]
    else:
        xs = [comp[xparam] for comp in result["state"]]

    if yparam in ctrl_param_name:
        ys = [comp[yparam] for comp in result["control"]]
    else:
        ys = [comp[yparam] for comp in result["state"]]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(xs, ys, s = 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    return fig, ax

def plot_scatter_feasibility(
    result:Dict,
    yparam:str,
    ylabel:str,
    ylims:List,
    filename: Optional[str] = None
):

    ys = np.array([comp[yparam] for comp in result["state"]])

    b_limits = np.array([comp["beta"] / comp["beta_troyon"] for comp in result["state"]])
    q_limits = np.array([comp["q_kink"] / comp["q"] for comp in result["state"]])
    n_limits = np.array([comp["n"] / comp["n_g"] for comp in result["state"]])
    f_limits = np.array([comp["f_BS"] / comp["f_NC"] for comp in result["state"]])

    # feasible solutions
    feasb_b_limit = np.array(result["b_limit"])
    feasb_q_limit = np.array(result["q_limit"])
    feasb_n_limit = np.array(result["n_limit"])
    feasb_f_limit = np.array(result["f_limit"])

    tau = np.array([comp["tau"] for comp in result["state"]])
    tbr = np.array([result["state"][idx]["TBR"] for idx in range(len(tau))])

    Qs = np.array([comp["Q"] for comp in result["state"]])

    feasb_indices = np.where(((feasb_b_limit == 1) * (feasb_n_limit == 1) * (feasb_q_limit == 1) * (feasb_f_limit == 1) * (tbr >= 1) * (Qs > 10.0)) == 1)
    feasb_indices = feasb_indices[0].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()

    yaxis = np.linspace(0.0, 20.0, 1000)

    axes[0].fill_betweenx(yaxis, 1, 5, facecolor="gray", alpha = 0.2)
    axes[0].scatter(b_limits, ys, s = 0.5)
    axes[0].scatter(b_limits[feasb_indices], ys[feasb_indices], c='r', s=0.75)
    axes[0].set_xlabel(r"$\beta / \beta_{Troyon}$")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlim([0,3.0])
    axes[0].set_ylim(ylims)

    axes[1].fill_betweenx(yaxis, 1, 5, facecolor="gray", alpha=0.2)
    axes[1].scatter(q_limits, ys, s=0.5)
    axes[1].scatter(q_limits[feasb_indices], ys[feasb_indices], c="r", s=0.75)
    axes[1].set_xlabel(r"$q_{kink}/q$")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlim([0, 2.5])
    axes[1].set_ylim(ylims)

    axes[2].fill_betweenx(yaxis, 1, 5, facecolor="gray", alpha=0.2)
    axes[2].scatter(n_limits, ys, s=0.5)
    axes[2].scatter(n_limits[feasb_indices], ys[feasb_indices], c="r", s=0.75)
    axes[2].set_xlabel(r"$\bar{n} / n_G$")
    axes[2].set_ylabel(ylabel)
    axes[2].set_xlim([0, 3.0])
    axes[2].set_ylim(ylims)

    axes[3].fill_betweenx(yaxis, 1, 5, facecolor="gray", alpha=0.2)
    axes[3].scatter(f_limits, ys, s=0.5)
    axes[3].scatter(f_limits[feasb_indices], ys[feasb_indices], c="r", s=0.75)
    axes[3].set_xlabel(r"$f_{op} / f_{bs}$")
    axes[3].set_ylabel(ylabel)
    axes[3].set_xlim([0, 3.0])
    axes[3].set_ylim(ylims)

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi = 120)

    plt.close()

    return fig, axes
