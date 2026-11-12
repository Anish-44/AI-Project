import random
import networkx as nx
from core.board import PCBProblem
from core.cost import count_crossings, compute_layers_needed, do_segments_intersect
from algorithms.random_restart import run_random_restart

def display_board(state, width, height):
    """Prints a visual representation of the board layout."""
    grid = [[" . " for _ in range(width)] for _ in range(height)]
    for comp, (x, y) in state.items():
        # Shorten the name slightly to fit the grid if needed
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
        layer_num = color + 1 # 0-indexed color to 1-indexed layer
        if layer_num not in layers:
            layers[layer_num] = []
        layers[layer_num].append(netlist[net_idx])
        
    return layers

def main():
    print("==================================================")
    print("      PCB COMPONENT PLACEMENT SOLVER BENCHMARK   ")
    print("==================================================\n")

    try:
        width = int(input("Enter board width: "))
        height = int(input("Enter board height: "))
        n = int(input("Enter number of components: "))
        
        # Hard constraint check before running
        if n > width * height:
            print(f"\n[ERROR] Result cannot be found. Cannot place {n} components on a {width}x{height} grid. Not enough space.")
            return

        components = [f"Comp{i}" for i in range(1, n + 1)]
        
        hot_input = input(f"Enter hot components by number (1 to {n}, comma separated, e.g. 1,3): ")
        hot_indices = [i.strip() for i in hot_input.split(",") if i.strip().isdigit()]
        hot_components = {f"Comp{i}" for i in hot_indices if int(i) <= n}

        num_nets = int(input(f"Enter number of random wire connections to generate: "))
        
        # Generate valid random netlist
        netlist = []
        max_possible_nets = (n * (n - 1)) // 2
        num_nets = min(num_nets, max_possible_nets) # Prevent infinite loops
        
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
        w_layers=15
    )

    print("\n--- Problem Configuration ---")
    print(f"Components: {', '.join(problem.components)}")
    print(f"Hot Components: {', '.join(problem.hot_components) if problem.hot_components else 'None'}")
    print(f"Netlist: {problem.netlist}\n")

    restarts = 30
    print(f"Running Random-Restart Hill Climbing ({restarts} restarts)...")
    results = run_random_restart(problem, num_restarts=restarts)

    if results['best_state'] is None:
        print("\n[ERROR] Result cannot be found. The algorithm failed to generate a valid layout.")
        return

    print("\n---------------- RESULTS ----------------")
    print(f"Algorithm:           {results['algorithm']}")
    print(f"Final Lowest Cost:   {results['final_cost']}")
    print(f"Total Evaluated:     {results['moves_evaluated']} moves")
    print(f"Execution Time:      {results['execution_time']:.5f} seconds")
    
    final_crossings = count_crossings(results['best_state'], problem.netlist)
    final_layers_count = compute_layers_needed(results['best_state'], problem.netlist)
    
    print(f"Wire Crossings:      {final_crossings}")
    print(f"Layers Used:         {final_layers_count}")
    
    print("\nLayer Breakdown:")
    layer_mapping = get_layer_breakdown(results['best_state'], problem.netlist)
    for layer_num in sorted(layer_mapping.keys()):
        nets = layer_mapping[layer_num]
        net_strings = [f"{n[0]}-{n[1]}" for n in nets]
        print(f"  Layer {layer_num}: {', '.join(net_strings)}")

    print("\nComponent Coordinates:")
    for comp, pos in results['best_state'].items():
        print(f"  - {comp:7s} : {pos}")

    display_board(results['best_state'], problem.width, problem.height)
    print("==================================================")

if __name__ == "__main__":
    main()  