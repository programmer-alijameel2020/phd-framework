class GeneticEncoder:
    def __init__(self, node={}, GENERATION=None):
        self.node = node
        self.layers = []
        self.GPTree = []
        self.GENERATION = GENERATION

    # Node structure
    def nodeStructure(self):
        self.node = {"measures": [], "genetics": [], "structure": [self.layers]}
        for node in self.GENERATION:
            self.GPTree.append(node)
