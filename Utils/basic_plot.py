import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_mean_std(data):
    mean_lst = []
    std_lst = []

    dim_num = len(data)

    for i in range(dim_num):
        data_i = data[:i + 1]
        mean_lst.append(data_i.mean())
        std_lst.append(data_i.std())

    y_upper = np.array(mean_lst) - np.array(std_lst)
    y_lower = np.array(mean_lst) + np.array(std_lst)

    x = np.linspace(1, dim_num, dim_num)
    fig, ax = plt.subplots(figsize=(10, 5))

    plt.plot(x, mean_lst, c='r')
    ax.fill_between(x, y_upper, y_lower, alpha=0.3, color='red')
    plt.xlabel('Num of Episodes')
    plt.ylabel('Reward')
    # plt.ylim(0, 1000)
