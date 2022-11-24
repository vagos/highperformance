import math
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

def make_heatmap(filename):

    with open(filename) as file:
        data = file.read().strip()

    data = re.findall(r'(\d+\.?\d+)', data)

    data = list(map(float, data))

    N = int(math.sqrt(len(data)))

    data = np.resize(data, (N, N))

    step = re.findall(r'(\d+)', filename)[0]

    plt.clf()
    plt.axis('off')
    sn.heatmap(data=data, cmap="coolwarm")
    plt.savefig(f"plots/density_step_{step.rjust(20, '0')}.png")
    
    print(f"Created plot for {filename}.")

for file in glob.glob("./plots/data/*.dat"):
    make_heatmap(file)
