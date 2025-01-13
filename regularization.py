
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
        return self.alpha * penalty
    
    # l2 penalty Function'
    def l2Penalty(self, weights):
        penalty = np.sum(weights ** 2)
        return self.alpha * penalty
    
    # Elastic Net Penalty Function
    def elasticNetPenalty(self, weights):
        l1Comp = self.l1Ratio * self.l1Penalty(weights)
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
        
# Application Use Case
if __name__ == "__main__":
    # Synth weights
    weights = np.array([0.5, -1.2, 3.0, 0.0, -0.7])
    
    # L1 Regularization
    l1Regz = Regularization(regz_type="l1", alpha=0.1)
    l1Penal = l1Regz.penaltyChoice(weights)
    print(f"L1 Penalty: {l1Penal}")
    
    # L2 Regularization
    l2Regz = Regularization(regz_type="l2", alpha=0.1)
    l2Penal = l2Regz.penaltyChoice(weights)
    print(f"L2 Penalty: {l2Penal}")
    
    # Elastic Net Regularization
    elasticNetRegz = Regularization(regz_type="elasticNet", alpha=0.1, l1Ratio=0.7)
    elasticNetPenal = elasticNetRegz.penaltyChoice(weights)
    print(f"Elastic Net Penalty: {elasticNetPenal}")