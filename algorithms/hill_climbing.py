from core.cost import compute_cost
from core.moves import get_neighbors

def steepest_ascent(initial_state, problem):
    current_state=initial_state
    current_cost=compute_cost(current_state, problem)
    evaluations=0

    while True:
        neighbors=get_neighbors(current_state, problem)
        best_neighbor=None
        best_cost=current_cost

        for n in neighbors:
            evaluations+=1
            cost=compute_cost(n, problem)
            if cost<best_cost:
                best_cost=cost
                best_neighbor=n

        if best_neighbor is None:
            break

        current_state=best_neighbor
        current_cost=best_cost

    return current_state, current_cost, evaluations