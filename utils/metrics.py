import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statistics import mean
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
LOGGER = logging.getLogger('MetricLogger')


def standard_metrics(probs, labels, *args, **kwargs):
    if len(probs.shape) == 1 and torch.all(torch.logical_or(labels == 0, labels == 1)):
        return standard_metrics_binary(probs, labels, *args, **kwargs)
    else:
        raise ValueError("[!] ERROR: labels are not binary!!")


def standard_metrics_binary(probs, labels, threshold=0.5, add_aucroc=True, add_optimal_acc=False, **kwargs):
    """
    Given predicted probabilities and labels, returns the standard metrics of accuracy, recall, precision, F1 score and AUCROC.
    The threshold, above which data points are considered to be classified as class 1, can also be adjusted.
    Returned values are floats (no tensors) in the dictionary 'metrics'.
    Probabilities and labels are expected to be pytorch tensors.
    """
    # YOUR CODE HERE:  write code to calculate accuracy, precision, recall, F1 and auroc. Return in the dictionary 'metrics'
    metrics = {}
    preds = (probs>threshold).int()

    metrics["accuracy"] = (preds == labels).float().mean()
    tp = ((preds==1) & (labels==1)).sum().float()
    fp = ((preds==0) & (labels==1)).sum().float()
    fn = ((preds==1) & (labels==0)).sum().float()
    metrics["precision"] = tp/(tp + fp)
    metrics["recall"] = tp/(tp + fn)

    metrics["f1"] = 2 * metrics["precision"] *  metrics["recall"] / (metrics["precision"] + metrics["recall"])

    if add_aucroc:
        metrics["aucroc"] = aucroc(probs, labels)

    return metrics



# OPTIONAL:  you can also optimize the cut-off threshold for the binary classification task (default=0.5)
def find_optimal_threshold(probs, labels, metric="accuracy"):
    """
    Given predicted probabilities and labels, returns the optimal threshold to use for the binary classification.
    It is conditioned on a metric ("accuracy", "F1", ...). Probabilities and labels are expected to be pytorch tensors.
    """
    # YOUR CODE HERE:  write code to find the best_threshold from a range of tested ones optimizing for the given metric
    
    return best_threshold


def aucroc(probs, labels):
    """
    Given predicted probabilities and labels, returns the AUCROC score used in the Facebook Meme Challenge.
    Inputs are expected to be pytorch tensors (can be cuda or cpu)
    """
    # YOUR CODE HERE:  compute the aucroc_score and return it
    labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    return roc_auc_score(labels, probs)


if __name__ == '__main__':

    num_classes = 2
    # generate probs
    probs = torch.randn(size=(1000,num_classes))
    probs = F.softmax(probs, dim=-1)
    # generate labels
    labels = (torch.multinomial(probs, num_samples=1).squeeze() > 0.5).long()
    print("Metrics", standard_metrics(probs[:,1], labels))