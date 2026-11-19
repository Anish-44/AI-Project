# algorithms/steepest_ascent.py

from core.board import generate_random_state
from core.cost import compute_cost
from core.moves import get_neighbors


def steepest_ascent_hill_climbing(problem, initial_state=None, max_iterations=1000):
    """
    Steepest Ascent Hill Climbing algorithm to optimize PCB component placement.
    Minimizes cost by taking the steepest descent approach and tracks total
    neighbor moves evaluated.
    """
    if initial_state is None:
        current_state = generate_random_state(problem)
    else:
        current_state = initial_state

    current_cost = compute_cost(current_state, problem)
    iterations = 0
    evaluations = 0

    while iterations < max_iterations:
        neighbors = get_neighbors(current_state, problem)

        if not neighbors:
            break

        best_neighbor = None
        best_neighbor_cost = float("inf")

        for neighbor in neighbors:
            evaluations += 1
            neighbor_cost = compute_cost(neighbor, problem)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        # Hit a local optimum
        if best_neighbor_cost >= current_cost:
            break

        current_state = best_neighbor
        current_cost = best_neighbor_cost
        iterations += 1

    return current_state, current_cost, evaluations


if __name__ == "__main__":
    from core.board import PCBProblem

    problem = PCBProblem()
    best_layout, final_cost, evals = steepest_ascent_hill_climbing(problem)

    print(f"Optimized Layout: {best_layout}")
    print(f"Final Cost: {final_cost}")
    print(f"Moves Evaluated: {evals}")