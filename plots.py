from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
plt.style.use("seaborn")

def plot_data(args, duration):
    np.random.seed(2)
    colors1 = ["red", "blue", "green", "indigo", "royalblue", "yellow", "peru", "palegreen"]
    colors2 = [(np.random.random(), np.random.random(), np.random.random()) for x in range(20)]
    COLORS = colors1 + colors2
    duration = duration
    '''
    if len(args) == 1:
        for key in args:
            plot_method = args[key]["mthd"]
            data = args[key]["data"]
            title = args[key]["title"]
            x_label = args[key]["x_label"]
            y_label = args[key]["y_label"]
            if plot_method == plt.plot:
                for p in range(data.shape[0]):
                    plt.plot(data[p], color=COLORS[p])
            elif plot_method == plt.eventplot:
                plt.eventplot(data, color=COLORS[:len(data)])
                plt.ylim([-0.5, len(data) - 0.5])
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
    else:
        '''
    plot, subs = plt.subplots(len(args),1)
    for key, sub in zip(args,subs):
        plot_method = args[key]["mthd"]
        data = args[key]["data"]
        title = args[key]["title"]
        x_label = args[key]["x_label"]
        y_label = args[key]["y_label"]
        if plot_method == plt.plot:
            for p in range(data.shape[0]):
                sub.plot(data[p], color=COLORS[p])
            sub.set_xlim([0, len(data[0])])
        elif plot_method == plt.eventplot:
            sub.eventplot(data, color=COLORS[:len(data)])
            sub.set_ylim([-0.5, len(data) - 0.5])
            sub.set_xlim([0, duration / 1000])
        sub.set_title(title)
        sub.set_xlabel(x_label)
        sub.set_ylabel(y_label)
    plt.show()

def plot_connectivity(connections):
    fig, (sub1, sub2) = plt.subplots(1,2)
    source = connections[0]
    target = connections[1]
    delays = False
    try:
        delays = connections[2]
    except:
        delays = np.ones(len(source))
    print(source)
    print(target)
    sub1.plot(np.zeros(len(source)), source, "ob")
    sub1.plot(np.ones(len(target)), target, "or")
    for i,j,d in zip(source,target,delays):
        sub1.plot([0,1],[i,j], "-k")
        sub1.text(i/2,(j+1)/2, d)
    sub1.set_yticks(list(range(max(len(set(source)), len(set(target))))))
    sub1.set_xticks([0,1])
    sub1.set_xticklabels(["Source", "Target"])
    plt.show()


def plot_connectivity_nx(connections):
    source = connections[0]
    target = connections[1]
    delays = False
    try:
        delays = connections[2]
    except:
        delays = np.ones(len(source))

    g = nx.DiGraph()
    label = {}
    for i,j, d in zip(source, target, delays):
        g.add_edge(i,j)
        label[(i,j)] = str(d)
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True,  connectionstyle='arc3, rad = 0.6')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=label)