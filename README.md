# Design Optimization of Tokamak Fusion Reactor through Deep Reinforcement Learning
## Introduction
This is a git repository of python codes for designing an optimal tokamak fusion reactors through deep reinforcement learning. Based on the reference, it is possible to determine the optimal plasma and design parameters, which satisfies conditions for steady state operation. We implemented our own reactor design computation code based on Friedberg's paper. You can check the validity of the tokamak design and performance through this computation code. 

<div>
    <p float = 'left'>
        <img src="/figures/design-computation-scheme.png"  width="75%">
    </p>
</div>

Our computation code can also provide the designed reactor performance, like lawson condition and the minimum requirements for steady-state operation, as given below.

<div>
    <p float = 'left'>
        <img src="/results/reference_lawson.png"  width="47.5%">
        <img src="/results/reference_overall.png" width="47.5%">
    </p>
</div>

The framework for optimizing the design configuration of a tokamak is based on proximal policy optimization. As the scheme given below, the reactor design computation code acts as an environment for interacting with the agent, which determines the input design parameters required for computing plasma parameters. The training process of the agents equals to the optimization process of a tokamak reactor design. In this research, however, it is required to design specific reward functions for inducing the agent to avoid operational limits. In our case, the minimum requirements for steady-state operation are represented as tanh function of (1-x) formula, which can prevent the parameters being close to limits. 

<div>
    <p float = 'left'>
        <img src="/figures/design-optimization-scheme.png"  width="75%">
    </p>
</div>

## How to execute
### Computation of the desired tokamak design
- Exectue main.py with modifying the configuration of the device in config/device_info.py
- You can print the detail of the tokamak design and its performance from Lawson criteria
    ```
        python3 main.py --save_dir {directory name} --tag {tag name} --use_benchmark {True or False} --use_rl {True or False}
    ```
### Find the optimal tokamak design with reinforcement learning
- We use Gridserach and Deep Reinforcement learning algorithm for reactor design optimization.
- In the case of RL, we use modified PPO algorithm to find out the optimal tokamak design which satisfies operation limits
- We designed the optimal tokamak which achieves high energy confinement while satisfying minimum cost (=size of tokamak).
- You can execute the design optmization codes as below.

1. Gridsearch
    ```
        python3 find_gridsearch.py --blanket_type {'solid' or 'liquid'} --num_episode {# of episodes} --n_grid {# of grids}
    ```

2. Bayesian optimization
    ```
        python3 find_bo.py --blanket_type {'solid' or 'liquid'} --num_episode {# of episodes} --init_random {# for random sampling}
    ```

3. Deep Reinforcement Learning
- Single CPU version
    ```
        python3 find_drl.py --blanket_type {'solid' or 'liquid'} --num_episode {# of episodes} --buffer_size {buffer size} --lr {learning rate}
    ```

- Multiple CPU version (Parallelization)
    ```
        python3 find_drl_parallel.py --blanket_type {'solid' or 'liquid'} --num_episode {# of episodes} --buffer_size {buffer size} --lr {learning rate} --n_workers {number of workers}
    ```

## Design optimization result based on RL

The figures below represent the comparison between original designed tokamak (left) and optimal designed tokamak based on DRL (right). We use the initial configuration of the tokamak from the grid search algorithm to find out the naive solution which satisfies the conditions for operation limit. Using single-step PPO algorithm, we can obtain the optimal design configuration of the tokamak which satisfies both the conditions and minimum cost condition, despite of some reduction in Q factor. 

- Lawson criteria 
<div>
    <p float = 'left'>
        <img src="/results/lawson_comparison.png"  width="75%">
    </p>
</div>

- Overall performance (Left:reference, Right:PPO)
<div>
    <p float = 'left'>
        <img src="/results/project_overall.png"  width="47.5%">
        <img src="/results/ppo_overall.png"  width="47.5%">
    </p>
</div>

- Design configuration (Left:reference, Right:PPO)
<div>
    <p float = 'left'>
        <img src="/results/project_poloidal_design.png"  width="47.5%">
        <img src="/results/ppo_poloidal_design.png"  width="47.5%">
    </p>
</div>

## 📖 Citation
If you use this repository in your research, please cite the following:

### 📜 Research Article
[Design Optimization of Nuclear Fusion Reactor through Deep Reinforcement Learning](https://doi.org/10.48550/arXiv.2409.08231)  
Jinsu Kim, Jaemin Seo, Arxiv, 2024.

### 📌 Code Repository
Jinsu Kim (2024). **Fusion-Reactor-Design-Optimization**. GitHub.  
[https://github.com/ZINZINBIN/Fusion-Reactor-Design-Optimization](https://github.com/ZINZINBIN/Fusion-Reactor-Design-Optimization)

#### 📚 BibTeX:
```bibtex
@software{Kim_RL_based_Fusion_2024,
author = {Kim, Jinsu},
doi = {https://doi.org/10.48550/arXiv.2409.08231},
license = {MIT},
month = sep,
title = {{Deep Reinforcement Learning based Fusion Reactor Design Optimization Code}},
url = {https://github.com/ZINZINBIN/Fusion-Reactor-Design-Optimization},
howpublished = {\url{https://github.com/ZINZINBIN/Fusion-Reactor-Design-Optimization}},
version = {1.0.0},
year = {2024}
}
```