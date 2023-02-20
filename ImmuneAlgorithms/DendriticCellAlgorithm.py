# Copyrights to @ali jameel al-mousawi
# Dendritic Cell Algorithm programming in python
import random
import array as ar


class DCA:
    def __init__(self):
        self.p_anomaly = 0.70
        self.p_normal = 0.95
        self.max_iter = 100
        self.num_cells = 10
        self.threshold = [5, 15]

    def randomBounce(self, min, max):
        return min + ((max-min) * random.Random())

    def randomVector(self, search_space):
        spaceArray = []
        for i in search_space:
            spaceArray.append(self.randomBounce(
                search_space[i][0], search_space[i][1]))

    def constructPattern(self, class_label, domain, p_safe, p_danger):
        # an array that contain the domain set
        domainSet = domain[class_label]
        randomSelection = random.Random(len(domainSet))
        patterns = {}  # the pattern array
        patterns["class_label"] = class_label
        patterns["inputs"] = domainSet[randomSelection]
        patterns["safe"] = random.Random() * p_safe * 100
        patterns["danger"] = random.Random() * p_danger * 100
        return patterns

    def generate_pattern(self, domain, p_anomaly, p_normal, prob_create_anom=0.5):
        pattern = ''
        if random.Random() < prob_create_anom:
            pattern = self.constructPattern(
                "Anomaly", domain, 1.0-p_normal, p_anomaly)
        else:
            pattern = self.constructPattern(
                "Normal", domain, p_normal, 1.0-p_anomaly)
        return pattern

    def cell_initialization(self, threshold, cell=[]):
        cell["lifeSpan"] = 1000.0
        cell["K"] = 0.0
        cell["cms"] = 0.0
        cell["migrationThreshold"] = self.randomBounce(
            threshold[0], threshold[1])
        cell["antigen"] = []
        return cell

    def exposeCell(self, cell, cms, k, pattern, threshold):
        cell["cms"] += cms
        cell["k"] += k
        cell["lifeSpan"] -= cms
        self.storeAntigen(cell, pattern["input"])
        if cell["lifeSpan"] <= 0:
            self.cell_initialization(threshold, cell)

    def storeAntigen(self, cell, input):
        if cell["antigen"][input]:
            cell["antigen"][input] = 1
        else:
            cell["antigen"][input] += 1

    def canCellMigrated(self, cell):
        if cell["cms"] >= cell["migrationThreshold"] and not cell["antigen"]:
            return cell

    def exposeAllCells(self, cells, pattern, threshold):
        migrate = []
        cms = pattern["safe"]+pattern["danger"]
        k = pattern["danger"]-pattern["safe"] * 2.0
        for x in cells:
            self.exposeCell(cells, cms, k, pattern, threshold)
            if self.canCellMigrated(cells[x]):
                migrate << cells[x]
                if cells["k"] > 0:
                    cells["class_label"] = "Anomaly"
                else:
                    cells["class_label"] = "Normal"
        return migrate

    def trainSystem(self, domain, max_iter, num_cell, p_anomaly, p_normal, threshold):
        immatureCells = []
        immatureCells[num_cell] = self.cell_initialization(threshold)
        migrated = []
        for iter in max_iter.times:
            pattern = self.generate_pattern(domain, p_anomaly, p_normal)
            migrants = self.exposeAllCells(immatureCells, pattern, threshold)
            for cell in migrants:
                immatureCells.pop(migrants[cell])
                immatureCells << self.cell_initialization(threshold)
                migrated << cell
        return migrated

    def classifyPattern(self, migrated, pattern):
        input = pattern["input"]
        num_cells, num_antiges = 0, 0
        for x in migrated:
            if(migrated[x]["class_label"] == "Anomaly" and not migrated[x]["antigen"][input]):
                num_cells += 1
                num_antiges += migrated[x]["antigen"][input]
        mcav = float(num_cells) / float(num_antiges)
        if mcav > 0.5:
            return "Anomaly"
        else:
            return "Normal"

    def testSystem(self, migrated, domain, p_anomaly, p_normal, num_trail=100):
        correctNorm = 0
        for x in num_trail:
            pattern = self.constructPattern(
                "Normal", domain, p_normal, 1.0-p_anomaly)
            class_label = self.classifyPattern(migrated, pattern)
            if class_label == "Normal":
                correctNorm += 1

        correct_anom = 0
        for times in num_trail:
            pattern = self.constructPattern(
                "Anomaly", domain, 1.0-p_normal, p_anomaly)
            class_label = self.classifyPattern(migrated, pattern)
            if class_label == "Anomaly":
                correct_anom += 1

        return [correct_anom, correctNorm]

    def execute(self, domain, max_iter, num_cells, p_anom, p_norm, threshold):
        migrated = self.trainSystem(
            domain, max_iter, num_cells, p_anom, p_norm, threshold)
        self.testSystem(migrated, domain, p_anom, p_norm)
        return migrated