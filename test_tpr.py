import numpy as np
from sklearn.metrics import roc_curve

def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find the largest FPR that is <= target_fpr
    idx = np.where(fpr <= target_fpr)[0][-1]
    return tpr[idx]

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(tpr_at_fpr(y_true, y_scores, 0.5))
