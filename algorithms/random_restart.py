import time
from core.board import generate_random_state
from algorithms.hill_climbing import steepest_ascent

def run_random_restart(problem, num_restarts=20):
    best_global_state=None
    best_global_cost=float('inf')
    total_evaluations=0
    restart_history=[]

    start_time=time.perf_counter()

    for r in range(1, num_restarts+1):
        start_state=generate_random_state(problem)
        local_state, local_cost, evals=steepest_ascent(start_state, problem)
        total_evaluations+=evals

        restart_history.append((r, local_cost, evals))

        if local_cost<best_global_cost:
            best_global_cost=local_cost
            best_global_state=local_state

    elapsed_time=time.perf_counter()-start_time

    return{
        "algorithm": f"Random_Restart Hill Climbing (k={num_restarts})",
        "final_cost": best_global_cost,
        "moves_evaluated": total_evaluations,
        "execution_time": elapsed_time,
        "best_state": best_global_state,
        "history": restart_history
    }