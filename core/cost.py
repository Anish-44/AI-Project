import networkx as nx

def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def do_segments_intersect(p1, p2, p3, p4):
    #to check whether intersection is there or not
    def ccw(A, B, C):
        return (C[1] - A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    if len({p1, p2, p3, p4})<4:
        return False

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def count_crossings(state, netlist):
    crossings=0
    num_nets=len(netlist)
    for i in range(num_nets):
        for j in range(i+1, num_nets):
            c1_a, c1_b = netlist[i]
            c2_a, c2_b = netlist[j]

            if len({c1_a, c1_b, c2_a, c2_b}) < 4:
                continue

            if do_segments_intersect(state[c1_a], state[c1_b], state[c2_a], state[c2_b]):
                crossings+=1
    return crossings

def compute_layers_needed(state, netlist):
    G=nx.Graph()
    num_nets=len(netlist)
    for i in range(num_nets):
        G.add_node(i)

    for i in range(num_nets):
        for j in range(i+1, num_nets):
            c1_a, c1_b = netlist[i]
            c2_a, c2_b = netlist[j]
            if len({c1_a, c1_b, c2_a, c2_b}) < 4:
                continue
            if do_segments_intersect(state[c1_a], state[c1_b], state[c2_a], state[c2_b]):
                G.add_edge(i, j)

    if G.number_of_edges()==0:
        return 1

    coloring=nx.coloring.greedy_color(G, strategy="largest_first")
    return max(coloring.values(), default=0)+1

def compute_cost(state, problem):
    wire_cost=sum(manhattan_distance(state[c1], state[c2])
    for c1, c2 in problem.netlist
    if c1 in state and c2 in state)

    thermal_cost=0
    hot_list=[c for c in problem.components if c in problem.hot_components and c in state]
    for i in range(len(hot_list)):
        for j in range(i+1, len(hot_list)):
            c1, c2 = hot_list[i], hot_list[j]
            if manhattan_distance(state[c1], state[c2])<problem.thermal_threshold:
                thermal_cost+=problem.thermal_penalty

    crossings=count_crossings(state, problem.netlist)
    crossings_cost=problem.w_crossings*crossings

    layers=compute_layers_needed(state, problem.netlist)
    layer_cost=problem.w_layers*layers

    return wire_cost + thermal_cost + crossings_cost + layer_cost