from collections import deque
import time
import tracemalloc

from core.cost import compute_cost
from core.moves import get_neighbors


def state_key(state):
    return tuple(sorted(state.items()))


def bfs_search(problem, initial_state, max_nodes=None, max_depth=None,
               time_limit=None, track_memory=True):
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()

    initial_cost = compute_cost(initial_state, problem)
    best_state, best_cost = dict(initial_state), initial_cost

    queue = deque([(initial_state, 0)])
    visited = {state_key(initial_state)}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier = 1
    max_depth_reached = 0
    cost_history = [(0, initial_cost)]
    stop_reason = "exhausted"

    while queue:
        if max_nodes is not None and nodes_expanded >= max_nodes:
            stop_reason = "max_nodes"
            break
        if time_limit is not None and time.perf_counter() - t0 >= time_limit:
            stop_reason = "time_limit"
            break

        state, depth = queue.popleft()
        nodes_expanded += 1

        if max_depth is not None and depth >= max_depth:
            continue

        for child in get_neighbors(state, problem):
            k = state_key(child)
            if k in visited:
                continue
            visited.add(k)
            nodes_generated += 1
            child_depth = depth + 1
            if child_depth > max_depth_reached:
                max_depth_reached = child_depth

            c = compute_cost(child, problem)
            if c < best_cost:
                best_state, best_cost = child, c
                cost_history.append((nodes_expanded, best_cost))

            queue.append((child, child_depth))

        max_frontier = max(max_frontier, len(queue))

    runtime = time.perf_counter() - t0
    peak_kb = None
    if track_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kb = peak / 1024

    return {
        "algorithm": "Breadth-First Search (BFS)",
        "best_state": best_state,
        "best_cost": best_cost,
        "final_cost": best_cost,
        "moves_evaluated": nodes_generated,
        "execution_time": runtime,
        "initial_cost": initial_cost,
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "max_frontier": max_frontier,
        "visited_size": len(visited),
        "max_depth_reached": max_depth_reached,
        "runtime_sec": runtime,
        "peak_memory_kb": peak_kb,
        "stop_reason": stop_reason,
        "cost_history": cost_history,
    }
