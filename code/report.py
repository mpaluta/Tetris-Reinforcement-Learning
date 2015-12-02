import matplotlib.pyplot as plt
import argparse
import sys

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

def read_file(fn):
    r = []
    d = []
    with open(fn) as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.startswith("INFO:root:DELTA:"):
                d.append(float(tokens[-1]))
            if line.startswith("INFO:root:REWARD:"):
                r.append(float(tokens[-1]))
    return (r,d)

def save_results(fn,outprefix):
    r_win = 25000
    d_win = 500

    r,d = read_file(sys.argv[1])
    winrx,winry = windowed_average(r, r_win)
    windx,windy = windowed_average(d, d_win)
    
    plt.plot(windx,windy)
    plt.ylabel("Delta")
    plt.xlabel("Timestep") 
    plt.title("Trailing average of {} latest delta values".format(d_win))
    plt.savefig("{}.delta.png".format(outprefix))
    plt.clf()

    plt.plot(winrx,winry)
    plt.ylabel("Reward")
    plt.xlabel("Timestep")
    plt.title("Trailing average of {} latest reward values".format(r_win))
    plt.savefig("{}.reward.png".format(outprefix))
    plt.clf()


def main():
    save_results(sys.argv[1], sys.argv[2])



if __name__ == "__main__":
    main()
