import pandas as pd
import os


class saveResult:

    def __init__(self, path):
        self.path = path
        self.bssSet = []

    def addBSS(self, BSS):
        self.bssSet.append(BSS)

    def saveCSV(self, result: list, path: str):
        path += '/result.csv'
        sc = pd.DataFrame(result, columns=['reward', 'success', 'failure', 'sensing'])
        sc.to_csv(path, index=False)

    def model_save(self):
        num = 1
        path = 'savedModel/scenario_4/model_%d' % num
        while os.path.exists(path):
            num += 1
            path = 'savedModel/scenario_4/model_%d' % num
        self._main_dnn.save(path)
        self.saveCSV(self._saveCSV, path)