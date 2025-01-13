
# Trying to create regularization functions from scratch.
import pandas as pd

class Regularization:
    # alpha = regularization strength nd ration is the ration of l1 to l2 in an elasic net case.
    def __init__(self, regz_type="l2", alpha=0.1, l1_ratio=0.5):
        self.regz_type = regz_type
        self.alpha = alpha
        self.l1_ration = l1_ratio