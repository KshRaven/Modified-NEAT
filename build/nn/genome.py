
from build.config import Config
from build.util.conversion import float_to_str

from numba import types, typeof, njit, optional, prange
from numba.experimental import jitclass
from numba.typed import List, Dict

import numpy as np

FLOAT   = types.float64
INT     = types.int64
STR     = typeof('str')
BOOL    = types.boolean


@jitclass([
    ('key', INT),
    ('bias', FLOAT),
])
class Node(object):
    def __init__(self, key: int):
        self.key  = key
        self.bias = 1.0

    def initialize(self, init_type: str, mean: float, std: float, min_: float, max_: float):
        if init_type == 'normal':
            self.bias = clamp(normal(mean, std), min_, max_)
        elif init_type == 'uniform':
            min_value = max(min_, (mean - (2 * std)))
            max_value = min(max_, (mean + (2 * std)))
            self.bias = uniform(min_value, max_value)
        else:
            raise ValueError(f"Unsupported NEAT Genome init type '{init_type}'")

    def distance(self, node: 'Node', compatibility_weight_coefficient: float):
        d = abs(self.bias - node.bias)
        return d * compatibility_weight_coefficient

    def copy(self):
        new_node = Node(self.key)
        new_node.bias = self.bias
        return new_node

    def crossover(self, node2: 'Node'):
        """ Creates a new node randomly inheriting attributes from its parents."""
        assert self.key == node2.key
        
        new_node = Node(self.key)
        if np.random.rand() > 0.5:
            new_node.bias = self.bias
        else:
            new_node.bias = node2.bias

        return new_node

    def mutate(self, mutate_rate: float, mutate_power: float, min_: float, max_: float,
               replace_rate: float, init_type: str, mean: float, std: float):
        r = np.random.random()
        if r < mutate_rate:
            self.bias = clamp(self.bias + normal(0.0, mutate_power), min_, max_)

        if r < replace_rate + mutate_rate:
            self.initialize(init_type, mean, std, min_, max_)

    def __str__(self):
        return f"Node{self.key}<bias={float_to_str(self.bias)}>"
    

@jitclass([
    ('key', types.Tuple([INT, INT])),
    ('weight', FLOAT),
])
class Connection(object):
    def __init__(self, key: tuple[int, int]):
        self.key   = key
        self.weight = np.random.normal(0, 1)

    def initialize(self, init_type: str, mean: float, std: float, min_: float, max_: float):
        if init_type == 'normal':
            self.weight = clamp(normal(mean, std), min_, max_)
        elif init_type == 'uniform':
            min_value = max(min_, (mean - (2 * std)))
            max_value = min(max_, (mean + (2 * std)))
            self.weight = uniform(min_value, max_value)
        else:
            raise ValueError(f"Unsupported NEAT Genome init type '{init_type}'")

    def distance(self, connection: 'Connection', compatibility_weight_coefficient: float):
        d = abs(self.weight - connection.weight)
        return d * compatibility_weight_coefficient

    def copy(self):
        new_conn = Connection(self.key)
        new_conn.weight = self.weight
        return new_conn

    def crossover(self, connection2: 'Connection'):
        """ Creates a new connection randomly inheriting attributes from its parents."""
        assert self.key == connection2.key
        
        new_conn = Connection(self.key)
        if np.random.rand() > 0.5:
            new_conn.weight = self.weight
        else:
            new_conn.weight = connection2.weight

        return new_conn

    def mutate(self, mutate_rate: float, mutate_power: float, min_: float, max_: float,
               replace_rate: float, init_type: str, mean: float, std: float):
        r = np.random.random()
        if r < mutate_rate:
            self.weight = clamp(self.weight + normal(0.0, mutate_power), min_, max_)

        elif r < replace_rate + mutate_rate:
            self.initialize(init_type, mean, std, min_, max_)

    def __str__(self):
        i, o = self.key
        return f"Connection({i}, {o})<weight={float_to_str(self.weight)}>"


@njit
def _required_for_output(inputs: list[int], outputs: list[int], connections: list[tuple[int, int]]):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """

    required = set(outputs)
    s = set(outputs)
    while True:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set([a for (a, b) in connections if b in s and a not in s])

        if not t:
            break

        layer_nodes = set([x for x in t if x not in inputs])
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


@njit
def _all(array: list[bool]):
    for idx in prange(len(array)):
        if array[idx] is False:
            return False
    return True


@njit
def _feed_forward_layers(input_keys: list[int], output_keys: list[int], connections: list[tuple[int, int]]):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param input_keys: list of the network input nodes
    :param output_keys: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = _required_for_output(input_keys, output_keys, connections)

    layers: list[list[int]] = List()
    s = set(input_keys)
    while True:
        # Find candidate nodes c for the next layer. These nodes should connect a node in s to a node not in s.
        c = set([b for (a, b) in connections if a in s and b not in s])
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and _all([a in s for (a, b) in connections if b == n]):
                t.add(n)

        if not t:
            break

        layer: list[int] = List.empty_list(INT)
        for n in t:
            layer.append(n)
        layers.append(layer)
        s = s.union(t)

    return layers


@njit
def zip_lb(layers: list[list[int]], build: list[tuple[int, int]]):
    unrolling = []
    for idx in range(len(layers)):
        unrolling.append((layers[idx], build[idx]))
    return unrolling


NODE = Node.class_type.instance_type
CONN = Connection.class_type.instance_type
CONN_TUPLE = types.Tuple((INT, INT))
LAYER = types.ListType(INT)
BUILD = types.Tuple([INT, INT])


@jitclass([
    ('key', INT),
    ('input_keys', types.ListType(INT)),
    ('output_keys', types.ListType(INT)),
    ('nodes', types.DictType(INT, NODE)),
    ('connections', types.DictType(CONN_TUPLE, CONN)),
    ('layers', types.ListType(types.ListType(INT))),
    ('inp_num', INT),
    ('out_num', INT),
    ('hidden_layers_num', types.ListType(INT)),
    ('build', types.ListType(BUILD)),
    ('node_indexer', INT),
])
class Network(object):
    def __init__(self, key: int, inputs: int, outputs: int, hidden_layers: list[int] = None, setup=True):
        # Set descriptors
        self.key        = key
        self.inp_num    = inputs
        self.out_num    = outputs
        hidden: list[int] = List.empty_list(INT)
        if hidden_layers is not None:
            for val in hidden_layers:
                hidden.append(val)
        self.hidden_layers_num = hidden
        self.input_keys: list[int] = List([key for key in prange(-inputs, 0)])
        self.output_keys: list[int] = List([key for key in prange(outputs)])

        # Create input and output nodes
        self.nodes: dict[int, Node] = Dict.empty(INT, NODE)
        if setup:
            for key in prange(-inputs, outputs):
                self.nodes[key] = Node(key)

        # Create hidden nodes when applicable
        # TODO: Create a way to initialize hidden nodes and layers an
        self.node_indexer = 0 if len(self.nodes) == 0 else max(List(self.nodes.keys()))
        # TODO: Create a way to initialize connections for hidden nodes and layers

        # Create connections based on initial connectivity type
        self.connections: dict[tuple[int, int], Connection] = Dict.empty(CONN_TUPLE, CONN)
        if setup:
            for out_key in prange(outputs):
                for inp_key in prange(-inputs, 0):
                    key = (inp_key, out_key)
                    self.connections[key] = Connection(key)

        self.layers: list[list[int]]      = List.empty_list(LAYER)
        self.build: list[tuple[int, int]] = List.empty_list(BUILD)

        # Run initial update
        if setup:
            self.update()

    def update(self):
        # Get layers
        self.layers: list = _feed_forward_layers(self.input_keys, self.output_keys, List(self.connections.keys()))

        # Get build
        build = List.empty_list(BUILD)
        inputs = self.inp_num
        for layer in self.layers:
            outputs = len(layer)
            build.append((inputs, outputs))
            inputs = outputs
        self.build = build

    @property
    def weights(self):
        array = []
        for layer, shape in zip_lb(self.layers, self.build):
            weight = np.ones(shape, np.float32)
            for x, node_key in enumerate(layer):
                for y, value in enumerate([conn.weight for (_, ok), conn in self.connections.items()
                                           if ok == node_key]):
                    weight[y, x] = value
            array.append(weight)
        return array

    @property
    def biases(self):
        return [
            np.array([self.nodes[node_key].bias for node_key in layer]) for layer in self.layers
        ]

    def raw(self):
        new_network = Network(self.key, self.inp_num, self.out_num, self.hidden_layers_num, False)
        return new_network

    def copy(self):
        new_network             = Network(self.key, self.inp_num, self.out_num, self.hidden_layers_num, False)
        new_network.nodes       = Dict([(key, node.copy()) for key, node in self.nodes.items()])
        new_network.connections = Dict([(key, conn.copy()) for key, conn in self.connections.items()])
        new_network.input_keys  = self.input_keys.copy()
        new_network.output_keys = self.output_keys.copy()
        new_network.node_indexer = self.node_indexer
        new_network.layers      = self.layers.copy()
        new_network.build       = self.build.copy()
        return new_network

    def __str__(self):
        if self.layers is not None:
            out = f"Network~{self.key} {'{'}"
            out += f"\n\tl{0}: ["
            for x, node_key in enumerate(self.input_keys):
                out += f"{node_key}"
                if x != len(self.input_keys)-1:
                    out += ", "
            out += "]"
            for idx, layer in enumerate(self.layers):
                out += f"\n\tl{idx+1}: ["
                for x, node_key in enumerate(layer):
                    out += f"{node_key}"
                    if x != len(layer)-1:
                        out += ", "
                out += "]"
            out += "\n}"
            return out
        else:
            return "Network not set"


@njit
def creates_cycle(connections: list[tuple[int, int]], test: tuple[int, int]):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


NETWORK = Network.class_type.instance_type


@jitclass([
    ('key', INT),
    ('networks', types.DictType(INT, NETWORK)),
    ('fitness', optional(FLOAT)),
])
class Genome(object):
    def __init__(self, key: int):
        self.key = key
        self.networks: dict[int, Network] = Dict.empty(INT, NETWORK)
        self.fitness: float = None

    def add_network(self, network_key: int, inputs: int, outputs: int, hidden_layers: list[int] = None):
        if network_key not in self.networks:
            network = Network(network_key, inputs, outputs, hidden_layers, True)
            self.networks[network_key] = network
        else:
            network = self.networks[network_key]
        return network

    def get_build(self, network_idx: int):
        return self.networks[network_idx].build

    def update_from_build(self):
        mapping = List(self.networks.keys())
        for idx in prange(len(mapping)):
            self.networks[mapping[idx]].update()

    def update_from_crossover(self, genome1: 'Genome', genome2: 'Genome'):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness is None or genome2.fitness is None:
            raise ValueError(f"A parent genome fitness has not been set")

        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Create and update networks
        mapping = List(parent1.networks.keys())
        for idx in prange(len(parent1.networks)):
            net_key = mapping[idx]
            network1 = parent1.networks[net_key]
            network2 = parent1.networks[net_key]
            network_clone = network1.raw()
            self.networks[network_clone.key] = network_clone

            # Inherit connection genes
            for key, conn1 in network1.connections.items():
                conn2 = network2.connections.get(key)
                if conn2 is None:
                    # Excess or disjoint gene: copy from the fittest parent.
                    self.networks[net_key].connections[key] = conn1.copy()
                else:
                    # Homologous gene: combine genes from both parents.
                    self.networks[net_key].connections[key] = conn1.crossover(conn2)

            # Inherit node genes
            parent1_set = network1.nodes
            parent2_set = network2.nodes
            for key, node1 in parent1_set.items():
                node2 = parent2_set.get(key)
                assert key not in self.networks[net_key].nodes
                if node2 is None:
                    # Extra gene: copy from the fittest parent
                    self.networks[net_key].nodes[key] = node1.copy()
                else:
                    # Homologous gene: combine genes from both parents.
                    self.networks[net_key].nodes[key] = node1.crossover(node2)

            # Update and set network
            self.networks[net_key].update()
        # for network in parent1.networks.values():
        #     for connection in network.connections.values():
        #         if connection.key not in parent2.networks[network.key].connections:
        #             if connection.weight != self.networks[network.key].connections[connection.key].weight:
        #                 raise ValueError(f"Incorrect breeding")

    def mutate(self, ssm: bool, na: float, nd: float, ca: float, cd: float,
               weight_params: tuple, bias_params: tuple):
        """ Mutates this genome. """
        # TODO: Finish coding the mutation methods

        mapping = List(self.networks.keys())
        for idx in prange(len(self.networks)):
            network = self.networks[mapping[idx]]

            # # For single structural mutation
            # if ssm:
            #     div = max(1, (na + nd + ca + cd))
            #     r   = np.random.rand()
            #     if r < (na/div):
            #         self.mutate_add_node(config)
            #     elif r < ((na + nd)/div):
            #         self.mutate_delete_node(config)
            #     elif r < ((na + nd + ca)/div):
            #         self.mutate_add_connection(config)
            #     elif r < ((na + nd + ca + cd)/div):
            #         self.mutate_delete_connection()
            # else:
            #     if np.random.rand() < na:
            #         self.mutate_add_node(config)
            #
            #     if np.random.rand() < nd:
            #         self.mutate_delete_node(config)
            #
            #     if np.random.rand() < ca:
            #         self.mutate_add_connection(config)
            #
            #     if np.random.rand() < cd:
            #         self.mutate_delete_connection()

            # Mutate connection genes.
            for connection in network.connections.values():
                connection.mutate(*weight_params)

            # Mutate node genes (bias, response, etc.).
            for node in network.nodes.values():
                node.mutate(*bias_params)

    # def add_connection(self, input_key, output_key, weight):
    #     # TODO: Add further validation of this connection addition?
    #     assert output_key >= 0
    #     key = (input_key, output_key)
    #     connection = Connection(key)
    #     connection.weight = weight
    #     self.connections[key] = connection
    #
    # def mutate_add_node(self):
    #     if len(self.connections) == 0:
    #         self.mutate_add_connection(config)
    #     else:
    #         # Choose a random connection to split
    #         conn_to_split = List(self.connections.values())[np.random.randint(len(self.connections))]
    #         self._last_node_index += 1
    #         new_node_id = self._last_node_index
    #         self.nodes[new_node_id] = Node(new_node_id)
    #
    #         # Disable this connection and create two new connections joining its nodes via
    #         # the given node.  The new node+connections have roughly the same behavior as
    #         # the original connection (depending on the activation function of the new node).
    #         conn_to_split.weight = 0.0
    #
    #         i, o = conn_to_split.key
    #         self.add_connection(i, new_node_id, 1.0)
    #         self.add_connection(new_node_id, o, conn_to_split.weight)
    #
    # def mutate_add_connection(self):
    #     """
    #     Attempt to add a new connection, the only restriction being that the output
    #     node cannot be one of the network input pins.
    #     """
    #     possible_outputs = list(self.nodes.keys())
    #     out_node = np.random.choice(np.array(possible_outputs))
    #
    #     possible_inputs = possible_outputs + list(self.network.input_keys)
    #     in_node = np.random.choice(possible_inputs)
    #
    #     # Don't duplicate connections.
    #     key = (in_node, out_node)
    #     if key in self.connections.keys():
    #         return
    #
    #     # Don't allow connections between two output nodes
    #     if in_node in self.network.output_keys and out_node in self.network.output_keys:
    #         return
    #
    #     # No need to check for connections between input nodes:
    #     # they cannot be the output end of a connection (see above).
    #
    #     # For feed-forward networks, avoid creating cycles.
    #     if creates_cycle(list(self.connections.keys()), key):
    #         return
    #
    #     cg = self.create_connection(config, in_node, out_node)
    #     self.connections[cg.key] = cg
    #
    # def mutate_delete_node(self, config):
    #     # Do nothing if there are no non-output nodes.
    #     available_nodes = [k for k in iterkeys(self.nodes) if k not in config.output_keys]
    #     if not available_nodes:
    #         return -1
    #
    #     del_key = choice(available_nodes)
    #
    #     connections_to_delete = set()
    #     for k, v in iteritems(self.connections):
    #         if del_key in v.key:
    #             connections_to_delete.add(v.key)
    #
    #     for key in connections_to_delete:
    #         del self.connections[key]
    #
    #     del self.nodes[del_key]
    #
    #     return del_key
    #
    # def mutate_delete_connection(self):
    #     if self.connections:
    #         key = choice(list(self.connections.keys()))
    #         del self.connections[key]

    def distance(self, other: 'Genome', compatibility_weight_coefficient: float, compatibility_disjoint_coefficient: float):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Run through each network
        total_distance = 0.0
        mapping = List(self.networks.keys())
        for idx in prange(len(mapping)):
            network = self.networks[mapping[idx]]
            other_network = other.networks[mapping[idx]]

            # Compute node gene distance component.
            node_distance = 0.0
            if network.nodes or other_network.nodes:
                disjoint_nodes = 0
                for k2 in other_network.nodes.keys():
                    if k2 not in network.nodes:
                        disjoint_nodes += 1

                for k1, n1 in network.nodes.items():
                    n2 = other_network.nodes.get(k1)
                    if n2 is None:
                        disjoint_nodes += 1
                    else:
                        # Homologous genes compute their own distance value.
                        node_distance += n1.distance(n2, compatibility_weight_coefficient)

                max_nodes = max(len(network.nodes), len(other_network.nodes))
                node_distance = (node_distance + (compatibility_disjoint_coefficient * disjoint_nodes)) / max_nodes

            # Compute connection gene differences.
            connection_distance = 0.0
            if network.connections or other_network.connections:
                disjoint_connections = 0
                for k2 in other_network.connections.keys():
                    if k2 not in network.connections:
                        disjoint_connections += 1

                for k1, c1 in network.connections.items():
                    c2 = other_network.connections.get(k1)
                    if c2 is None:
                        disjoint_connections += 1
                    else:
                        # Homologous genes compute their own distance value.
                        connection_distance += c1.distance(c2, compatibility_weight_coefficient)

                max_conn = max(len(network.connections), len(other_network.connections))
                connection_distance = (connection_distance + (compatibility_disjoint_coefficient * disjoint_connections)) / max_conn

            network_distance = node_distance + connection_distance
            total_distance += network_distance
        return total_distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        nodes_num = sum([len(net.nodes) for net in self.networks.values()])
        connections_num = sum([len(net.connections) for net in self.networks.values()])
        return nodes_num, connections_num


@njit
def clamp(value: float, minimum: float, maximum: float):
    return max(min(value, maximum), minimum)


@njit
def normal(mean: float, std: float):
    return np.random.normal(mean, std)


@njit
def uniform(a: float, b: float):
    return a + (b - a) * np.random.randn()


@njit
def _init_nodes(genome: Genome, init_type: str, mean: float, std: float, min_: float, max_: float):
    for network in genome.networks.values():
        for node in network.nodes.values():
            node.initialize(init_type, mean, std, min_, max_)


@njit
def _init_connections(genome: Genome, init_type: str, mean: float, std: float, min_: float, max_: float):
    for network in genome.networks.values():
        for connection in network.connections.values():
            connection.initialize(init_type, mean, std, min_, max_)


def initialize_genome(genome: Genome, config: Config):
    _init_nodes(genome, config.genome.init_type,
                config.genome.bias_init_mean, config.genome.bias_init_std,
                config.genome.bias_min_value, config.genome.bias_max_value)
    _init_connections(genome, config.genome.init_type,
                      config.genome.weight_init_mean, config.genome.weight_init_std,
                      config.genome.weight_min_value, config.genome.weight_max_value)


def load_network(network: Network, struct: Dict):
    network.key                = struct['key']
    network.inp_num            = struct['inp_num']
    network.out_num            = struct['out_num']
    network.hidden_layers_num  = List.empty_list(INT)
    for i in struct['hidden_layers_num']:
        network.hidden_layers_num.append(i)
    network.input_keys         = List(struct['input_keys'])
    network.output_keys        = List(struct['output_keys'])
    nodes = Dict.empty(INT, NODE)
    for key, bias in struct['nodes']:
        n = Node(key)
        n.bias = bias
        nodes[key] = n
    connections = Dict.empty(CONN_TUPLE, CONN)
    for key, weight in struct['connections']:
        c = Connection(key)
        c.weight = weight
        connections[key] = c
    network.nodes              = nodes
    network.connections        = connections
    network.layers             = List([List(i) for i in struct['layers']])
    network.build              = List(struct['build'])


def load_genome(struct: Dict):
    key             = struct['key']
    genome          = Genome(key)
    genome.fitness  = struct['fitness']
    networks        = Dict.empty(INT, NETWORK)
    for i, ns in enumerate(struct['networks']):
        network = Network(i, 1, 1)
        load_network(network, ns)
        networks[network.key] = network
    genome.networks  = networks
    return genome
