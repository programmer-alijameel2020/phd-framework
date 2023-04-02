class GeneticEncoder:
    def __init__(self, node={}, GENERATION=None):
        self.node = node
        self.layers = []
        self.GPTree = []
        self.GENERATION = GENERATION

    # Node structure Measures [] : is an array that holds the quality metrics where the parameters [accuracy, f1,
    # precision, recall, loss, validation_accuracy, validation_f1, validation_precision, validation_recall,
    # validation_loss, mean squared error, learning rate]

    # genetics []: holds the genetics parameter like [generations, fitness, crossover rate, mutation rate]
    # structure [{"layer_type":"", "params":{}}]


    def nodeStructure(self):
        self.node = {"measures": [], "genetics": [], "structure": [self.layers]}
        for node in self.GENERATION:
            self.GPTree.append(node)
