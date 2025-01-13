
# Trying to create regularization functions from scratch.
import numpy as np

class Regularization:
    # alpha = regularization strength nd ration is the ration of l1 to l2 in an elasic net case.
    def __init__(self, regz_type="l2", alpha=0.1, l1Ratio=0.5):
        self.regz_type = regz_type
        self.alpha = alpha
        self.l1Ratio = l1Ratio
        
    # l1 Penalty Function
    def l1Penalty(self, weights):
        penalty = np.sum(np.abs(weights))
        return self.apha * penalty
    
    # l2 penalty Function'
    def l2Penalty(self, weights):
        penalty = np.sum(weights ** 2)
        return self.alpha * penalty
    
    # Elastic Net Penalty Function
    def elasticNetPenalty(self, weights):
        l1Comp = self.l1ratio * self.l1Penalty(weights)
        l2Comp = (1 - self.l1Ratio) * self.l2Penalty(weights)
        return l1Comp + l2Comp
    
    # Compute choiced Penalties from the ones above
    def penaltyChoice(self, weights):
        if self.regz_type == "l1":
            return self.l1Penalty(weights)
        elif self.regz_type == "l2":
            return self.l2Penalty(weights)
        elif self.regz_type == "elasticNet":
            return self.elasticNetPenalty(weights)
        else:
            raise ValueError("Invalid regularzation type. Pick(l1/ l2/ elasticNet)")