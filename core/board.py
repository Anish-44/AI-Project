import random

class PCBProblem:
    def __init__(
        self, width=4, height=4, 
        components=None,
        hot_components=None,
        netlist=None, thermal_threshold=3,
        thermal_penalty=10, w_crossings=3, 
        w_layers=15
    ):
        self.width=width
        self.height=height
        self.components=components if components else ["CPU", "PWR", "RAM", "CAP"]
        self.hot_components=set(hot_components) if hot_components else {"CPU", "PWR"}
        self.netlist=netlist if netlist else [("CPU", "RAM"), ("CPU", "CAP"), ("RAM", "CAP"), ("PWR", "CAP")]
        self.thermal_penalty=thermal_penalty
        self.thermal_threshold=thermal_threshold
        self.w_crossings=w_crossings  
        self.w_layers=w_layers

def generate_random_state(problem):
    occupied=set()
    state={}
    for comp in problem.components:
        while True:
            pos=(random.randint(0, problem.width-1), random.randint(0, problem.height-1))
            if pos not in occupied:
                occupied.add(pos)
                state[comp]=pos
                break

    return state