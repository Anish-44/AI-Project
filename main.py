import math
import random
import time
import tracemalloc
import networkx as nx

from core.board import PCBProblem, generate_random_state
from core.cost import compute_cost, count_crossings, compute_layers_needed, do_segments_intersect
from algorithms.bfs_search import bfs_search
from algorithms.steepest_ascent import steepest_ascent_hill_climbing
from algorithms.random_restart import run_random_restart
from algorithms.simulated_annealing import simulated_annealing


def calculate_bfs_time_limit(problem):
    """
    Computes an input-aware time limit (in seconds) for BFS based on:
    - Permutation state space size: P(width * height, num_components)
    - Evaluation complexity per state: O(num_nets^2)
    """
    grid_cells = problem.width * problem.height
    n = len(problem.components)
    num_nets = len(problem.netlist)

    total_states = math.perm(grid_cells, n)

    # Estimated seconds per node evaluation (calibrated to ~0.6 ms at 20 nets)
    est_sec_per_node = 0.00015 + 0.0000012 * (num_nets ** 2)
    est_full_runtime = total_states * est_sec_per_node

    # Small state spaces: allocate enough headroom to exhaust the entire search space
    if est_full_runtime <= 20.0:
        return round(max(5.0, min(25.0, est_full_runtime * 1.4)), 2)

    # Large state spaces: scale logarithmically with problem size, capped at 30s
    log_states = math.log10(max(10, total_states))
    scaled_limit = 8.0 + (log_states * 1.5) + (num_nets * 0.15)
    return round(min(30.0, max(10.0, scaled_limit)), 2)


def display_board(state, width, height):
    """Prints a visual representation of the board layout."""
    grid = [[" . " for _ in range(width)] for _ in range(height)]
    for comp, (x, y) in state.items():
        short_name = comp.replace("Comp", "C")
        grid[y][x] = f"{short_name:^3}"

    print("\nVisual Grid Layout:")
    print("+" + "---+" * width)
    for row in grid:
        print("|" + "|".join(row) + "|")
        print("+" + "---+" * width)


def get_layer_breakdown(state, netlist):
    """Builds the conflict graph and groups nets by their assigned physical layer."""
    G = nx.Graph()
    num_nets = len(netlist)
    for i in range(num_nets):
        G.add_node(i)

    for i in range(num_nets):
        for j in range(i + 1, num_nets):
            c1_a, c1_b = netlist[i]
            c2_a, c2_b = netlist[j]
            if len({c1_a, c1_b, c2_a, c2_b}) < 4:
                continue
            if do_segments_intersect(state[c1_a], state[c1_b], state[c2_a], state[c2_b]):
                G.add_edge(i, j)

    if G.number_of_edges() == 0:
        return {1: netlist}

    coloring = nx.coloring.greedy_color(G, strategy="largest_first")

    layers = {}
    for net_idx, color in coloring.items():
        layer_num = color + 1
        if layer_num not in layers:
            layers[layer_num] = []
        layers[layer_num].append(netlist[net_idx])

    return layers


def run_with_memory_tracking(func, *args, **kwargs):
    """Executes an algorithm while measuring runtime and peak memory in KB."""
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        output = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    return output, elapsed, peak_bytes / 1024.0


def print_algorithm_report(res, problem):
    """Prints detailed metrics, layer routing, coordinates, and grid for one algorithm."""
    state = res["best_state"]
    crossings = count_crossings(state, problem.netlist)
    layers_count = compute_layers_needed(state, problem.netlist)
    res["crossings"] = crossings
    res["layers_used"] = layers_count

    print("\n==================================================")
    print(f"ALGORITHM: {res['algorithm']}")
    print("==================================================")
    print(f"Final Lowest Cost:   {res['final_cost']}")
    print(f"Moves Evaluated:     {res['moves_evaluated']}")
    print(f"Execution Time:      {res['execution_time']:.5f} seconds")
    print(f"Peak Memory Used:    {res['peak_memory_kb']:.2f} KB")
    if "stop_reason" in res:
        print(f"Search Stop Reason:  {res['stop_reason']}")
    print(f"Wire Crossings:      {crossings}")
    print(f"Layers Used:         {layers_count}")

    print("\nLayer Breakdown:")
    layer_mapping = get_layer_breakdown(state, problem.netlist)
    for layer_num in sorted(layer_mapping.keys()):
        nets = layer_mapping[layer_num]
        net_strings = [f"{n[0]}-{n[1]}" for n in nets]
        print(f"  Layer {layer_num}: {', '.join(net_strings)}")

    print("\nComponent Coordinates:")
    for comp, pos in state.items():
        print(f"  - {comp:7s} : {pos}")

    display_board(state, problem.width, problem.height)

def evaluate_algorithms(all_results, w_cost=0.50, w_time=0.30, w_mem=0.20):
    """
    Ranks algorithms using Min-Max normalized Cost, Execution Time, and Peak Memory.
    Returns (best_result, ranked_results) where higher 'overall_score' (0-100) is better.
    """
    valid_results = [r for r in all_results if r.get("best_state") is not None]
    if not valid_results:
        return None, []

    costs = [r["final_cost"] for r in valid_results]
    times = [r["execution_time"] for r in valid_results]
    mems = [r["peak_memory_kb"] for r in valid_results]

    def normalize(val, min_v, max_v):
        return (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0

    min_c, max_c = min(costs), max(costs)
    min_t, max_t = min(times), max(times)
    min_m, max_m = min(mems), max(mems)

    for r in valid_results:
        norm_cost = normalize(r["final_cost"], min_c, max_c)
        norm_time = normalize(r["execution_time"], min_t, max_t)
        norm_mem = normalize(r["peak_memory_kb"], min_m, max_m)

        penalty = (w_cost * norm_cost) + (w_time * norm_time) + (w_mem * norm_mem)
        r["composite_penalty"] = penalty
        r["overall_score"] = round(100.0 * (1.0 - penalty), 2)

    ranked = sorted(valid_results, key=lambda r: r["overall_score"], reverse=True)
    return ranked[0], ranked


def main():
    print("==================================================")
    print("      PCB COMPONENT PLACEMENT SOLVER BENCHMARK   ")
    print("==================================================\n")

    try:
        width = int(input("Enter board width: "))
        height = int(input("Enter board height: "))
        n = int(input("Enter number of components: "))

        if n <= 0 or width <= 0 or height <= 0:
            print("\n[ERROR] Dimensions and component count must be positive integers.")
            return

        if n > width * height:
            print(f"\n[ERROR] Result cannot be found. Cannot place {n} components on a {width}x{height} grid. Not enough space.")
            return

        components = [f"Comp{i}" for i in range(1, n + 1)]

        hot_input = input(f"Enter hot components by number (1 to {n}, comma separated, e.g. 1,3): ")
        hot_indices = [i.strip() for i in hot_input.split(",") if i.strip().isdigit()]
        hot_components = {f"Comp{i}" for i in hot_indices if 1 <= int(i) <= n}

        num_nets = int(input("Enter number of random wire connections to generate: "))

        netlist = []
        max_possible_nets = (n * (n - 1)) // 2
        num_nets = min(max(0, num_nets), max_possible_nets)

        while len(netlist) < num_nets:
            c1, c2 = random.sample(components, 2)
            if (c1, c2) not in netlist and (c2, c1) not in netlist:
                netlist.append((c1, c2))

    except ValueError:
        print("\n[ERROR] Invalid input. Please enter integers where required.")
        return

    problem = PCBProblem(
        width=width,
        height=height,
        components=components,
        hot_components=hot_components,
        netlist=netlist,
        thermal_threshold=3,
        thermal_penalty=10,
        w_crossings=3,
        w_layers=15,
    )

    initial_state = generate_random_state(problem)
    initial_cost = compute_cost(initial_state, problem)
    total_state_space = math.perm(width * height, n)
    bfs_time_limit = calculate_bfs_time_limit(problem)

    print("\n--- Problem Configuration ---")
    print(f"Board Size:        {width}x{height} ({total_state_space:,} possible states)")
    print(f"Components:        {', '.join(problem.components)}")
    print(f"Hot Components:    {', '.join(sorted(problem.hot_components)) if problem.hot_components else 'None'}")
    print(f"Netlist ({len(problem.netlist)}):       {problem.netlist}")
    print(f"Initial Cost:      {initial_cost}")
    print(f"Adaptive BFS Cap:  {bfs_time_limit}s")

    all_results = []

    # 1. Breadth-First Search (BFS) - Adaptive time limit based on input scale
    print(f"\n[1/4] Running Breadth-First Search (BFS, limit={bfs_time_limit}s)...")
    bfs_res = bfs_search(
        problem,
        initial_state=dict(initial_state),
        max_nodes=None,
        max_depth=None,
        time_limit=bfs_time_limit,
        track_memory=True,
    )
    all_results.append(bfs_res)

    # 2. Steepest-Ascent Hill Climbing
    print("[2/4] Running Steepest-Ascent Hill Climbing...")
    (sa_hc_state, sa_hc_cost, sa_hc_evals), sa_hc_time, sa_hc_mem = run_with_memory_tracking(
        steepest_ascent_hill_climbing, problem, dict(initial_state)
    )
    sa_hc_res = {
        "algorithm": "Steepest-Ascent Hill Climbing",
        "final_cost": sa_hc_cost,
        "moves_evaluated": sa_hc_evals,
        "execution_time": sa_hc_time,
        "peak_memory_kb": sa_hc_mem,
        "best_state": sa_hc_state,
    }
    all_results.append(sa_hc_res)

    # 3. Random-Restart Hill Climbing
    restarts = 30
    print(f"[3/4] Running Random-Restart Hill Climbing (k={restarts})...")
    rr_res, _, rr_mem = run_with_memory_tracking(
        run_random_restart, problem, num_restarts=restarts
    )
    rr_res["peak_memory_kb"] = rr_mem
    all_results.append(rr_res)

    # 4. Simulated Annealing
    print("[4/4] Running Simulated Annealing...")
    sim_res, _, sim_mem = run_with_memory_tracking(simulated_annealing, problem)
    sim_res["peak_memory_kb"] = sim_mem
    all_results.append(sim_res)

    # Print detailed breakdown for every algorithm
    for res in all_results:
        if res["best_state"] is None:
            print(f"\n[ERROR] {res['algorithm']} failed to find a valid layout.")
            continue
        print_algorithm_report(res, problem)

    best_overall, ranked_results = evaluate_algorithms(
        all_results, w_cost=0.50, w_time=0.30, w_mem=0.20
    )

    # Side-by-side summary comparison table
    print("\n====================================================================================================================")
    print("                                      ALGORITHM COMPARISON SUMMARY                                                  ")
    print("====================================================================================================================")
    header = (
        f"{'Algorithm':<36} | {'Cost':>6} | {'Layers':>6} | {'Cross':>5} | "
        f"{'Evaluated':>10} | {'Time (s)':>9} | {'Memory (KB)':>11} | {'Score':>7}"
    )
    print(header)
    print("-" * len(header))

    for res in all_results:
        print(
            f"{res['algorithm']:<36} | "
            f"{res['final_cost']:>6} | "
            f"{res['layers_used']:>6} | "
            f"{res['crossings']:>5} | "
            f"{res['moves_evaluated']:>10} | "
            f"{res['execution_time']:>9.5f} | "
            f"{res['peak_memory_kb']:>11.2f} | "
            f"{res['overall_score']:>6.2f}"
        )
    print("====================================================================================================================")

    print(
        f"\nBest Overall Strategy (Cost + Time + Memory): {best_overall['algorithm']} "
        f"(Score: {best_overall['overall_score']}/100 | Cost: {best_overall['final_cost']}, "
        f"Time: {best_overall['execution_time']:.2f}s, Memory: {best_overall['peak_memory_kb']:.2f} KB)"
    ) 


if __name__ == "__main__":
    main()