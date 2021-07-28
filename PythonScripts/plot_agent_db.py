import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_dbs', nargs='+', help='List of directories to be plotted')
    parser.add_argument("-i", "--individual_plots", action="store_true", help="Will plot individual agent performance")
    parser.add_argument("-a", "--area_plots", action="store_true", help="Will plot area fill between min/max values within one db")
    parser.add_argument("-c", "--cm_name", type=str, default="nipy_spectral", help="Name of matplotlib colormap to plot with")
    parser.add_argument("-l", "--horizontal_line", type=int, help="Will plot a horizontal white line at this Elo value")
    parser.add_argument("--hl_name", type=str, help="Label to use for horizontal line")
    args = parser.parse_args()
    print(args)
    
    color_map = cm.get_cmap(args.cm_name)
    fig,ax = plt.subplots(1,1)
    
    all_avgs = []
    
    for i,agent_db in enumerate(args.agent_dbs):
        plot_color = color_map((i+1)/(len(args.agent_dbs)+1))
        
        first = True        
        c = 0
        for subdir, dirs, files in os.walk(agent_db):
            for dir in dirs:
                if dir[-1] != '/' or dir[-1] != '\\':
                    dir += '/'
                elo_log = np.load(subdir + dir + "elo_log.npy")
                c += 1
                
                if first:
                    first = False
                    mins = np.full(elo_log.shape, 9999, dtype=np.uint32)
                    maxs = np.zeros(elo_log.shape, dtype=np.uint32)
                    avgs = np.zeros(elo_log.shape, dtype=np.float32)
                    
                print(subdir, dir, "ELO", elo_log[-1])
                
                for j in range(elo_log.shape[0]):
                    val = elo_log[j]
                    avgs[j] += val
                    if mins[j] > val:
                        mins[j] = val
                    if maxs[j] < val:
                        maxs[j] = val
                        
                if args.individual_plots:
                    ax.plot(range(len(elo_log)), elo_log, color=plot_color, alpha=.25)
                        
        avgs /= c
        if args.area_plots:
            ax.fill_between(range(len(mins)), maxs, mins, where= maxs > mins, facecolor=plot_color, alpha=0.5, interpolate=True)
        all_avgs.append((avgs, plot_color, agent_db))
        
    for avgs,plot_color,agent_db in all_avgs:
        ax.plot(range(len(avgs)), avgs, color=plot_color, label=agent_db.strip('/'))
        
    if args.horizontal_line:
        ax.plot(range(len(all_avgs[0][0])), [args.horizontal_line for _ in range(len(all_avgs[0][0]))], color=color_map(0), label=args.hl_name if args.hl_name else "")
        
    leg = ax.legend(loc="lower left", prop={"size":6})
    ax.set_title("Average Elo Attained per Matchmaking Strategy (10 seeds each)")
    ax.set_xlabel("Steps (max 10M)")
    ax.set_ylabel("Average Elo")
    plt.show()