import math
import numpy as np
from typing import Union, Callable


class Node:
    """Stores state information as well as information important for the MCTS"""

    def __init__(self, state: np.ndarray, seed: int):
        """Constructor

        Args:
            state (np.ndarray): State to be encoded into a node.
            seed (int): seed for reproducible results
        """
        self.state = state
        self.id = hash(self.state)
        self.visits = 0
        self.total_state_action_value = 0
        self.prior = 0
        self.seed = seed

    def _set_priors(self, p: list[float], epsilon=0.25):
        """Sets priors with dirichlet noise

        Args:
            p (list[float]): prior probabilities for action-space
            epsilon (float, optional): Epsilon constant to scale dirichlet noise - see paper. Defaults to 0.25.
        """
        # add dirichlet noise
        # explanation https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
        ALPHA_CONSTANT = 0.3  # based on alphazero paper
        self.prior = (1 - epsilon) * p + epsilon * np.random.default_rng(
            seed=self.seed
        ).dirichlet([ALPHA_CONSTANT for _ in self.prior])

    def __str__(self):
        return f"This is node {self.id}"


class DirectedGraph:
    """Stores Game Tree"""

    def __init__(
        self,
        nodes: Union[None, list[Node]] = None,
        edges: Union[None, list[Node]] = None,
    ):
        """Constructor

        Args:
            nodes (Union[None,list[Node]], optional): Prepopulated nodes for game tree. Defaults to None.
            edges (Union[None,list[Node]], optional): Prepopulated nodes for game tree. Defaults to None.
        """
        self.nodes = nodes if nodes is not None else {}
        self.edges = edges if edges is not None else {}

    def add_node(self, node: Node):
        """Adds node to graph.

        Args:
            node (Node): Node to add
        """
        self.nodes[node.id] = node
        self.edges[node.id] = []

    def add_edge(self, node1: Node, node2: Node):
        """Adds directed edge from node1 to node2 to graph

        Args:
            node1 (Node): Source node for edge.
            node2 (Node): Destination node for edge.
        """
        self.edges[node1.id].append(node2)

    def get_connected_nodes(self, node: Node) -> dict[Node]:
        """Gets all decending edges from node

        Args:
            node (Node): Target node

        Returns:
            dict[Node]: All decending edges from node
        """
        return self.edges[node.id]


def uct(parent_node: Node, child_nodes: list[Node], c: float) -> Node:
    """Finds best node to select based of parent node by upper confidence bound for trees

    Args:
        parent_node (Node): Current Parent Node
        child_nodes (list[Node]): possible child nodes
        c (float): scaling factor

    Returns:
        Node: Best node based on uct measures
    """
    scores = []
    for node in child_nodes:
        # prior score
        score = c * node.prior * math.sqrt(parent_node.visits) / (node.visits + 1)
        # U(s,a) + Q(s,a)
        scores.append(score + (node.total_state_action_value / node.visits))

    idx = np.argmax(scores)
    return child_nodes[idx]


class MCTSSolver:
    def __init__(
        self,
        num_iterations: int,
        c: float,
        tau: float,
        epsilon: float,
        model,
        seed=1234,
        selection_strategy=uct,
        Graph=None,
    ):
        """Constructor

        Args:
            num_iterations (int): Number of iterations used for Monte Carlo Tree Search
            c (float): Constant used for selection strategy
            tau (float): Temperature parameter see paper
            epsilon (float): Epsilon for calculating priors
            model (torch.nn): Pytorch policy model
            seed (int, optional): Seed for reproducible results. Defaults to 1234.
            selection_strategy (_type_, optional): Used selection strategy function. Defaults to uct.
            Graph (DirectedGraph, optional): If available use pretrained graph . Defaults to None.
        """

        self.G = DirectedGraph() if Graph is None else Graph
        self.model = model
        self.num_iterations = num_iterations
        self.seed = seed

        self.c = c
        self.tau = tau
        self.epsilon = epsilon
        self.selection_strategy = selection_strategy

    def _select(
        self, node: Node, selection_strategy: Callable[[Node, Node, float], Node]
    ) -> Node:
        """Selects suitable node via selection strategy.

        Args:
            node (Node): currently selected root node
            selection_strategy (_type_): strategy to choose next nodes. Currently only uct is supported.

        Returns:
            Node: Best node by selection strategy
        """
        path = []
        while node.visits > 0:
            node = selection_strategy(
                node, self.G.get_connected_nodes(self.current_node), self.c
            )
            path.append(node)

        return node, path

    def _expand(self, node: Node, p: np.ndarray) -> None:
        """Generates new states from allowed actions based on model, sets priors and and adds them to the tree.

        Args:
            node (Node): Root node
            p (numpy.ndarray): prior probabilites for actions
        """
        actions = self.model.get_legal_actions(node.state)
        children = []

        # adds possible childen to Graph
        for action in actions:
            state, _ = self.model.step(node.state, action)
            child_node = Node(state, seed=self.seed)
            child_node._set_priors(self.model, p, self.epsilon)
            children.append(child_node)
            self.G.add_node(child_node)
            self.G.add_edge(node, child_node)

    def _evaluate(self, node) -> None:
        pass

    def _backup(self, path: list[Node], v: int) -> None:
        """Backpropagates value to all nodes in path

        Args:
            path (list[Node]): List of all nodes visited on the path.
            v (int): Reward signal to be propagated to all nodes in path.
        """
        for node in path:
            node.visits += 1
            node.total_state_action_value += v

    def _get_proba(self, node: Node) -> list[float]:
        """Converts visits and scores of mcts to probability distribution for action-space

        Args:
            node (Node): Root node for which to calculate the probability distribution.

        Returns:
            list[float]: Probability distribution of action-space
        """
        N_a = [
            self.G.nodes[next_node.id].visits
            for next_node in self.get_connected_nodes(node)
        ]
        probas = []
        for selected_action in N_a:
            numerator = selected_action.visits ** (1 / self.tau)
            denominator = sum(action.visits ** (1 / self.tau) for action in N_a)
            probas.append(numerator / denominator)
        return probas

    def solve(self, node: Node) -> list[float]:
        """Calcuates probability distribution for action-space based on set numbers of iterations.

        Args:
            node (Node): Root Node

        Raises:
            ValueError: If terminal node is send to the mcts solver

        Returns:
            list[float]: Probability distribution of action-space for root node
        """
        # for inference set numbers of iterations to 0 #TODO: verify if thats all that is needed
        for _ in range(self.num_iterations):
            if self.model.is_terminal(node.state) == True:
                raise ValueError(
                    "Node is terminal node"
                )  # TODO: change this since this should probably handle terminal nodes gracefully

            selected_node, path = self._select(node, self.selection_strategy)
            v, p = self.agent.net(selected_node.state)
            self._expand(selected_node, p)
            self._backup(path, v)
        return self._get_probas(node)