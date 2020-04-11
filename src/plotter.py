import matplotlib.pyplot as plt
import seaborn as seabornInstance
import numpy as np


def seaborn_barchart_plot(series):
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(series)


def plot_bar(x_labels, series_1, series_2, y_label, title):
    x = np.arange(len(x_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, series_1, width, label='Men')
    rects2 = ax.bar(x + width / 2, series_2, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def plot_py_chart(sizes, labels, title):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.title(title)
    plt.tight_layout()

    plt.show()


def seaborn_scatter(x, y, title, x_label=None, y_label=None):
    seabornInstance.regplot(x=x, y=y, fit_reg=False).set_title(title)

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.show()


def buble_col(x, y, z, u, title, x_label, y_label):
    # Change color with c and alpha. I map the color to the X axis value.
    plt.scatter(x, y, s=z, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)

    # Add titles (main and on axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

