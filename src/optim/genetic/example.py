import numpy as np
from typing import Callable, List

def create_population(pop_size: int, dim: int, bounds):
    return np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))

# compute fitness
def fitness(pop: np.array, eval_f: Callable):
    return (-1) * np.array([eval_f(ind) for ind in pop])

# selection
def selection(pop: np.array, fit: np.array, num_parents: int):
    idx = np.argsort(fit)[-num_parents:]
    return pop[idx]

# tournament selection
def tournament_selection(pop: np.array, fit: np.array, num_parents: int):
    selected = []

    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), num_parents)
        best = idx[np.argmax(fit[idx])]
        selected.append(pop[best])

    return np.array(selected)


# X-over: swapping halves to create children from the parents
def crossover(parents, offspring_size: int):
    offspring = []

    for _ in range(offspring_size):
        p1, p2 = (
            parents[np.random.randint(len(parents))],
            parents[np.random.randint(len(parents))],
        )
        alpha = np.random.rand()
        child = alpha * p1 + (1 - alpha) * p2
        offspring.append(child)

    return np.array(offspring)


def mutation(offspring: np.ndarray, mutation_rate: float, bounds):

    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            dim = np.random.randint(offspring.shape[1])
            offspring[i, dim] += np.random.normal(0, 0.1)
            offspring[i, dim] = np.clip(offspring[i, dim], bounds[0], bounds[1])

    return offspring


# general algorithm for optimizing the function f
def genetic_algorithm(
    f: Callable,
    dim: int = 5,
    bounds: List = [-5.0, 5.0],
    pop_size: int = 64,
    generations: int = 128,
    mutation_rate: float = 0.2,
    num_parents: int = 32,
):
    pop = create_population(pop_size, dim, bounds)
    best_scores = []

    for gen in range(generations):
        fit = fitness(pop, f)
        best_scores.append(-fit.max())

        # parents = selection(pop, fit, num_parents)
        parents = tournament_selection(pop, fit, num_parents)

        offspring = crossover(parents, pop_size - num_parents)
        offspring = mutation(offspring, mutation_rate, bounds)

        pop = np.vstack([parents, offspring])
        print(f"Gen {gen+1}: Best f(x) = {best_scores[-1]:.6f}")

    best_ind = pop[np.argmax(fitness(pop, f))]
    return best_ind, f(best_ind), best_scores

if __name__ == "__main__":

    def test_f(x):
        return np.sum(x**2)

    best_x, best_fx, scores = genetic_algorithm(test_f)
    print("\nBest solution found:", best_x)
    print("Function value at best solution:", best_fx)