# Design Optimization of Tokamak Fusion Reactor through Deep Reinforcement Learning
## Introduction
This is a git repository of python codes for designing an optimal tokamak fusion reactors through deep reinforcement learning. Based on the reference, it is possible to determine the optimal plasma and design parameters, which satisfies conditions for steady state operation. We implemented our own reactor design computation code based on Friedberg's paper. You can check the validity of the tokamak design and performance through this computation code. 

<div>
    <p float = 'left'>
        <img src="/figures/design-computation-scheme.png"  width="640" height="400">
    </p>
</div>

Our computation code can also provide the designed reactor performance, like lawson condition and the minimum requirements for steady-state operation, as given below.

<div>
    <p float = 'left'>
        <img src="/results/reference_lawson.png"  width="360" height="320">
        <img src="/results/reference_overall.png"  width="360" height="320">
    </p>
</div>

The framework for optimizing the design configuration of a tokamak is based on proximal policy optimization. As the scheme given below, the reactor design computation code acts as an environment for interacting with the agent, which determines the input design parameters required for computing plasma parameters. The training process of the agents equals to the optimization process of a tokamak reactor design. In this research, however, it is required to design specific reward functions for inducing the agent to avoid operational limits. In our case, the minimum requirements for steady-state operation are represented as tanh function of (1-x) formula, which can prevent the parameters being close to limits. 

<div>
    <p float = 'left'>
        <img src="/figures/design-optimization-scheme.png"  width="640" height="400">
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

2. Deep Reinforcement Learning
    ```
        python3 find_drl.py --blanket_type {'solid' or 'liquid'} --num_episode {# of episodes} --buffer_size {buffer size} --lr {learning rate}
    ```

## Design optimization result based on RL

The figures below represent the comparison between original designed tokamak (left) and optimal designed tokamak based on DRL (right). We use the initial configuration of the tokamak from the grid search algorithm to find out the naive solution which satisfies the conditions for operation limit. Using single-step PPO algorithm, we can obtain the optimal design configuration of the tokamak which satisfies both the conditions and minimum cost condition, despite of some reduction in Q factor. 

- Lawson criteria 
<div>
    <p float = 'left'>
        <img src="/results/lawson_comparison.png"  width="600" height="480">
    </p>
</div>

- Overall performance (Left:reference, Right:PPO)
<div>
    <p float = 'left'>
        <img src="/results/project_overall.png"  width="360" height="320">
        <img src="/results/ppo_overall.png"  width="360" height="320">
    </p>
</div>

- Design configuration (Left:reference, Right:PPO)
<div>
    <p float = 'left'>
        <img src="/results/project_poloidal_design.png"  width="360" height="320">
        <img src="/results/ppo_poloidal_design.png"  width="360" height="320">
    </p>
</div>

## Reference
- Designing a tokamak fusion reactor : How does plasma physics fit in? (J.P.Freidberg, F.J.Mangiarotti, J.Minervini, Physics of plasmas, 2015)
- Spectial topics on fusion reactor enigneering, Jisung Kang, KFE
- Lecture note on Fusion Reactor Technology 1, Yong-su Na, SNU
- Direct shape optimization through deep reinforcement learning, Jonathan Viquerat et al

## Code Citation
```
    @ARTICLE{2017arXiv170504862C,
    author = {{Clemente}, A.~V. and {Castej{\'o}n}, H.~N. and {Chandra}, A.
        },
        title = "{Efficient Parallel Methods for Deep Reinforcement Learning}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1705.04862},
    primaryClass = "cs.LG",
    keywords = {Computer Science - Learning},
        year = 2017,
        month = may,
    adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170504862C},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
```