from math import sqrt, floor, ceil
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import json


### PLOT STATISTICS
def plot(path_to_results, threshold=0.5, save_fig=True):
    with open(path_to_results, "r") as f:
        metrics_results = json.load(f)
    
    if save_fig:
        result_path = path_to_results.replace(".json", ".png")

    factor = sqrt(len(metrics_results))
    ROWS = ceil(factor)
    COLS = floor(factor)

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(ROWS,COLS, figsize=(1920*px, 1080*px))
    fig.suptitle("Metrics statistics", fontsize="xx-large", fontweight='bold')

    for i, (metric, result) in enumerate(metrics_results.items()):
        x,y = i//ROWS, i%ROWS

        scores = np.array(result["scores"])
        if "hallucination" not in metric.lower() and "uncertain" not in metric.lower():
            passed, failed = scores[scores >= threshold], scores[scores < threshold]
        else:
            metric = metric.replace("Uncertainity", "Uncertainty") # typo hotfix
            passed, failed = scores[scores < threshold], scores[scores >= threshold]

        ax[x,y].hist([passed, failed], color=["green", "red"])#, edgecolor="black")
        ax[x,y].set_xlim((0.0, 1.0))
        ax[x,y].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[x,y].axvline(x = threshold, color = 'b', linestyle = "--", linewidth = 1)
        ax[x,y].set_title(metric)

    fig.legend(labels=["threshold", "passed", "failed"], ncols=3, fontsize="large", loc="upper right")
    
    if save_fig:
        plt.savefig(result_path)
    
    plt.show()
