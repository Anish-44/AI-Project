# algorithms/steepest_ascent.py

from core.board import generate_random_state
from core.cost import compute_cost
from core.moves import get_neighbors

def steepest_ascent_hill_climbing(problem, initial_state=None, max_iterations=1000):
    """
    Steepest Ascent Hill Climbing algorithm to optimize PCB component placement.
    Since cost.py calculates penalties, this minimizes the cost by taking the 
    steepest descent approach.
    """
    # Start with a given state or generate a random one using board.py
    if initial_state is None:
        current_state = generate_random_state(problem)
    else:
        current_state = initial_state
        
    current_cost = compute_cost(current_state, problem)
    iterations = 0
    
    while iterations < max_iterations:
        # Generate all possible neighbor configurations from moves.py
        neighbors = get_neighbors(current_state, problem)
        
        if not neighbors:
            break
            
        # Find the neighbor with the absolute best (lowest) cost
        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        for neighbor in neighbors:
            neighbor_cost = compute_cost(neighbor, problem)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                
        # If the best neighbor does not improve our current layout, we've hit a local optimum
        if best_neighbor_cost >= current_cost:
            break
            
        # Move to the best neighbor
        current_state = best_neighbor
        current_cost = best_neighbor_cost
        iterations += 1
        
    return current_state, current_cost

if __name__ == "__main__":
    # Quick local test to verify the algorithm runs
    from core.board import PCBProblem
    
    problem = PCBProblem()
    best_layout, final_cost = steepest_ascent_hill_climbing(problem)
    
    print(f"Optimized Layout: {best_layout}")
    print(f"Final Cost: {final_cost}")