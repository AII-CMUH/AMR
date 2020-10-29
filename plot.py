def get_data(fname):
    container = np.load('path/to/file')
    ms_data = [container[key] for key in container]
    ms_df = pd.DataFrame(ms_data).fillna(0)
    ms_df = ms_df.to_numpy()
    ms_label = pd.read_csv('path/to/file')
    
    return ms_label, ms_df


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)
      

def get_UBandLB(mean, std):
    ub = mean + std
    lb = mean - std
    return ub, lb


def ci095(data):
    confidence = 0.95
    n = len(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return h


def datainconditions(label, df):
    R_data = []
    S_data = []

    for i, rs in enumerate(label[label.columns[-1]]):
        if rs == 1:
            R_data.append(df[i])
        else:
            S_data.append(df[i])
    return R_data, S_data


def plotMSfig(ConName, R_data, S_data, s1, s2, ds):
    hR = ci095(R_data)
    meanR = np.average(R_data, 0)
    fig = plt.figure(figsize=(28, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    x_ticks = np.arange(s1, s2, ds)
    plt.xticks(x_ticks)
    ax.set_xlim(s1, s2)
    # plt.errorbar(range(2000, 2000+meanR.shape[0]), meanR, yerr=hR, color='red', ecolor='pink', label='MRSA')
    plt.errorbar(range(s1, s2), meanR[(s1-2000):(s2-2000)], yerr=hR[(s1-2000):(s2-2000)], color='red', ecolor='pink', label=ConName[0])


    hS = ci095(S_data)
    meanS = np.average(S_data, 0)
    # plt.errorbar(range(2000, 2000+meanS.shape[0]), meanS, yerr=hS, color='b', ecolor='lightblue', alpha=0.7, label='MSSA')
    plt.errorbar(range(s1, s2), meanS[(s1-2000):(s2-2000)], yerr=hS[(s1-2000):(s2-2000)], color='b', ecolor='lightblue', alpha=0.7, label=ConName[1])

    plt.legend(loc='upper right', prop={'size': 32})
