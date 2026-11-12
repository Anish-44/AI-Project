def get_neighbors(state, problem):
    neighbors=[]
    comp_names=list(state.keys())

    for i in range(len(comp_names)):
        for j in range(i+1, len(comp_names)):
            c1, c2=comp_names[i], comp_names[j]
            new_state=state.copy()
            new_state[c1], new_state[c2]=new_state[c2], new_state[c1]
            neighbors.append(new_state)

    occupied=set(state.values())
    directions=[(-1, 0), (1, 0), (0, -1), (0, 1)]
    for comp, (x, y) in state.items():
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0<=nx<problem.width and 0<=ny<problem.height:
                if (nx, ny) not in occupied:
                    new_state=state.copy()
                    new_state[comp]=(nx, ny)
                    neighbors.append(new_state)

    return neighbors