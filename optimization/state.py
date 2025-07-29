import networkx as nx

class State:
    """
    A class to represent a state in the greedy exploration process.
    Attributes:
        state (nx.MultiGraph): The Tanner graph representing the code.
        cost_result (dict): A dictionary containing the cost result of the state.
            'logical_error_rate' (float): The logical error rate of the state.
            'std' (float): The standard deviation of the logical error rate.
            'runtime' (float): The runtime of the decoding operation for this state.
    Methods:
        __init__(state, cost_result): Initializes the state with a Tanner graph and its cost result.
        __repr__(): Returns a string representation of the state.
        __str__(): Returns a string describing the state with its cost.
        """
    def __init__(self, state: nx.MultiGraph, cost_result: dict):
        self.state = state
        self.cost_result = cost_result

    def __repr__(self):
        return f"State(cost={self.cost_result['logical_error_rate']:.6f})"

    def __str__(self):
        return f"State with cost {self.cost_result['logical_error_rate']:.6f}"