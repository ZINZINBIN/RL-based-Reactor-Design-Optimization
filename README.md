# Project : Reactor Design and Analysis (Fusion Reactor Technology 1)
## Introduction
This is a git repository of python codes for designing fusion reactors. Based on the reference, it is possible to determine the plasma parameters and reactor design by using this code. If basic quantities are determined, the reactor parameters can be computed and it is able to check the validity of the tokamak design. 

<div>
    <p float = 'left'>
        <img src="/results/reference_lawson.png"  width="360" height="320">
        <img src="/results/reference_overall.png"  width="360" height="320">
    </p>
</div>

## How to execute
### Computation of the desired tokamak design
- Exectue main.py with modifying the configuration of the device in config/device_info.py
- You can print the detail of the tokamak design and its performance from Lawson criteria
    ```
        python3 main.py --save_dir {directory name} --tag {tag name}
    ```
### Find the optimal tokamak design with reinforcement learning
- We use single-step PPO algorithm to find out the optimal tokamak design which satisfies operation limits
- Our aim is to acheive high energy confinement by designing the optimal tokamak based on RL algorithm
    ```
        python3 find_optimal.py
    ```

## Design optimization result based on RL
We use the initial configuration of the tokamak from the grid search algorithm to find out the naive solution which satisfies the conditions for operation limit. Using single-step PPO algorithm, we can obtain the optimal design configuration of the tokamak which satisfies both the conditions and minimum cost condition. This optimal solution even satisfies that the performance is greater than Q = 10.
- Lawson criteria 
<div>
    <p float = 'left'>
        <img src="/results/project_lawson.png"  width="360" height="320">
        <img src="/results/ppo_lawson.png"  width="360" height="320">
    </p>
</div>

- Overall performance
<div>
    <p float = 'left'>
        <img src="/results/project_overall.png"  width="360" height="320">
        <img src="/results/ppo_overall.png"  width="360" height="320">
    </p>
</div>

## Reference
- Designing a tokamak fusion reactor : How does plasma physics fit in? (J.P.Freidberg, F.J.Mangiarotti, J.Minervini, Physics of plasmas, 2015)
- Spectial topics on fusion reactor enigneering, Jisung Kang, KFE
- Lecture note on Fusion Reactor Technology 1, Yong-su Na, SNU
- Direct shape optimization through deep reinforcement learning, Jonathan Viquerat et al