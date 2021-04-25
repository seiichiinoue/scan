import matplotlib.pyplot as plt

def learning_curve(path):
    lls, ppls, lls_v, ppls_v = [], [], [], []
    log = open(path).readlines()
    for line in log:
        if not line.startswith("iter:"):
            continue
        _, i, _, ll, _, ppl = line.split()
        lls.append(float(ll))
        ppls.append(float(ppl))
    plot_fig(lls, 'log_likelihood')
    plot_fig(ppls, 'perplexity')

def plot_fig(arr, s='perplexity'):
    # perplexity
    fig = plt.figure(figsize=(6.0, 4.0), dpi=200)
    plt.plot(range(len(arr)), arr, label='train', color='darkblue')
    plt.xlabel('iteration')
    plt.ylabel(s)
    plt.legend()
    plt.title(s)
    plt.savefig(f"./fig/{s}.png")

learning_curve("../out_transport")