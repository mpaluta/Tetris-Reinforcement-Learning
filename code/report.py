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
    absd = []
    with open(fn) as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.startswith("INFO:root:DELTA:"):
                d.append(float(tokens[-1]))
                absd.append(abs(float(tokens[-1])))
            if line.startswith("INFO:root:REWARD:"):
                r.append(float(tokens[-1]))
    return (r,d,absd)

def save_results(dirname):
    r_win = 10000
    d_win = 500
    absd_win = 500

    r,d,absd = read_file("{}/log".format(dirname))
    winrx,winry = windowed_average(r, r_win)
    windx,windy = windowed_average(d, d_win)
    winabsdx,winabsdy = windowed_average(absd, absd_win)
   
    outprefix = "{}/plots".format(dirname)
 
    plt.plot(windx,windy)
    plt.ylabel("Delta")
    plt.xlabel("Timestep (block placements)") 
    plt.title("Trailing average of {} latest delta values".format(d_win))
    plt.savefig("{}.delta.png".format(outprefix))
    plt.clf()

    plt.plot(winabsdx,winabsdy)
    plt.ylabel("Delta magnitude")
    plt.xlabel("Timestep (block placements)") 
    plt.title("Trailing average of {} latest delta magnitudes".format(absd_win))
    plt.savefig("{}.abs_delta.png".format(outprefix))
    plt.clf()

    plt.plot(winrx,winry)
    plt.ylabel("Reward")
    plt.xlabel("Timestep (game ticks)")
    plt.title("Trailing average of {} latest reward values".format(r_win))
    plt.savefig("{}.reward.png".format(outprefix))
    plt.clf()


def main():
    save_results(sys.argv[1])



if __name__ == "__main__":
    main()
