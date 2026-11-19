import math
import random
import time
from core.board import generate_random_state
from core.cost import compute_cost
from core.moves import get_neighbors

def simulated_annealing(
    problem,
    initial_temp=100.0,
    cooling_rate=0.995,
    min_temp=0.01,
    max_iterations=5000,
):
    """Hill climbing that sometimes accepts a worse neighbor.

    A random neighbor is always taken when it is cheaper. A more expensive
    neighbor is taken with probability exp(-cost_increase / temperature).
    Temperature drops each step, so late moves behave like plain hill climbing.
    The cheapest layout seen is kept even if a later worse move is accepted.
    """
    current_state = generate_random_state(problem)
    current_cost = compute_cost(current_state, problem)
    best_state = current_state
    best_cost = current_cost

    temperature = initial_temp
    evaluations = 0
    accepted_worse = 0
    history = [(0, current_cost, temperature)]

    start_time = time.perf_counter()

    for step in range(1, max_iterations + 1):
        if temperature < min_temp:
            break

        neighbors = get_neighbors(current_state, problem)
        if not neighbors:
            break

        neighbor = random.choice(neighbors)
        neighbor_cost = compute_cost(neighbor, problem)
        evaluations += 1
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            if delta >= 0:
                accepted_worse += 1
            current_state = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = current_state

        temperature *= cooling_rate
        if step % 100 == 0:
            history.append((step, best_cost, temperature))

    elapsed_time = time.perf_counter() - start_time

    return {
        "algorithm": "Simulated Annealing Hill Climbing",
        "final_cost": best_cost,
        "moves_evaluated": evaluations,
        "execution_time": elapsed_time,
        "best_state": best_state,
        "accepted_worse": accepted_worse,
        "history": history,
        "final_temperature": temperature,
    }
