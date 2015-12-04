import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import sys
import os

def windowed_average(a, wsize):
    results = []
    if not a:
        return [],[]
    s = sum(a[:wsize])
    xs = []
    for i in range(wsize-1,len(a)):
        xs.append(i)
        results.append(s / wsize)
        s -= a[i+1-wsize]
        if i < len(a)-1:
            s += a[i+1]
    return xs,results

def read_windowed_stats(fn, windows):
    r = []
    d = []
    absd = []

    with open(fn) as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.startswith("INFO:root:DELTA:"):
                d.append(float(tokens[-1]))
                absd.append(abs(float(tokens[-1])))
            if line.startswith("INFO:root:REWARD:"):
                r.append(float(tokens[-1]))
    return {
        "reward": windowed_average(r, windows["reward"]), 
        "delta": windowed_average(d, windows["delta"]),
        "abs_delta": windowed_average(absd, windows["abs_delta"])
        }

def make_plot(fn, xlabel, ylabel, title, stats, stat_name):
    #line_styles = ["r--", "b-.", "g:", "o-", "p:"]
    line_styles = [
        {
            "color": "blue"
        },
        {
            "color": "green"
        },
        {
            "color": "orange"
        },
        {
            "color": "red"
        },
        {
            "color": "pink"
        },
    ]
    for i,name in enumerate(stats.keys()):
        print "name: {},   stat_name={}".format(name,stat_name)
        xs,ys = stats[name][stat_name]
        kwargs = line_styles[i]
        plt.plot(xs,ys,label=name, **kwargs)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel) 
    plt.title(title)
    plt.legend(loc=3)
    plt.savefig(fn)
    plt.clf()
    

def plot_results(output_dir, input_dirs, windows):

    stats = OrderedDict()
    for input_dir in input_dirs:
        name = input_dir.split("/")[-1]
        log = "{}/log".format(input_dir)
        stats[name] = read_windowed_stats(log, windows)

    make_plot("{}/delta.png".format(output_dir), "Timestep (block placements)", "Delta", "Trailing average of {} latest delta values".format(windows["delta"]), stats, "delta")
    make_plot("{}/abs_delta.png".format(output_dir), "Timestep (block placements)", "Delta magnitude", "Trailing average of {} latest delta magnitudes".format(windows["abs_delta"]), stats, "abs_delta")
    make_plot("{}/reward.png".format(output_dir), "Timestep (game ticks)", "Reward", "Trailing average of {} latest reward values".format(windows["reward"]), stats, "reward")


def main():
    output_dir = "plots/{}".format(sys.argv[1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    windows = {"delta": 500,
               "abs_delta": 500,
               "reward": 30000}

    plot_results(output_dir, sys.argv[2:], windows)



if __name__ == "__main__":
    main()
