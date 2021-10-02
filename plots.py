from matplotlib import pyplot as plt
plt.style.use("seaborn")

def plot_data(args):
    all_args = ["v", "u", "spikes", "input"]
    loc_args = {}
    COLORS = ["red", "blue", "green", "indigo", "royalblue", "yellow", "peru", "palegreen"]
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
        elif plot_method == plt.eventplot:
            sub.eventplot(data, color=COLORS[:len(data)])
            sub.set_ylim([-0.5,len(data) - 0.5])
        sub.set_title(title)
        sub.set_xlabel(x_label)
        sub.set_ylabel(y_label)
    plt.show()

def plot_connectivity(connections):
    fig, (sub1, sub2) = plt.subplots(1,2)
    source = connections[0]
    target = connections[1]
    sub1.plot(source, range(len(source)), "ok")
    sub1.plot(source, range(len(target)), "ok")
    #sub1.set_xticks([0,1], ["Source", "Target"])