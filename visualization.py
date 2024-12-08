import numpy as numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import pandas as pd
from matplotlib import cm



def main():
    output = pd.read_csv("./output.csv")
    output_groupped = output.groupby("Time")

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    plots = {}
    time_values = sorted(output["Time"].unique())
    unique_bodies = sorted(output["Body"].unique())

    cmap = cm.get_cmap('tab10', len(unique_bodies))
    body_colors = {body: cmap(i) for i,body in enumerate(unique_bodies)}
    # for group_name, group_data in output_groupped:
    #     plots[group_name], = ax.plot([],[],[], marker = 'o', label = group_name)

    ax.set_xlim(output['Px'].min(), output['Px'].max())
    ax.set_ylim(output['Py'].min(), output['Py'].max())
    ax.set_zlim(output['Pz'].min(), output['Pz'].max())

    ax.legend()
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_zlabel("Z-coordinate")

    def update(frame):
       # ax.clear()
        snapshot = output_groupped.get_group(time_values[frame])
        ax.set_xlim(output['Px'].min(), output['Px'].max())
        ax.set_ylim(output['Py'].min(), output['Py'].max())
        ax.set_zlim(output['Pz'].min(), output['Pz'].max())
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_zlabel("Z-coordinate")

        for body in unique_bodies:
            body_data = snapshot[snapshot["Body"] == body]
            ax.scatter(body_data["Px"], body_data["Py"], body_data["Pz"], c =body_colors[body], marker = "o")

        
        # current_time = frame
        # for group_name, group_data in output_groupped:
        #     #print(group_data)
        #     current_data = group_data[group_data["Time"] == current_time]
        #     plots[group_name].set_data(current_data["Px"], current_data["Py"])
        #     plots[group_name].set_3d_properties(current_data["Pz"])
        # return plots.values()
    handles = [plt.Line2D ([], [], color = body_colors[body], marker = "o", label = f"Body {body}") for body in unique_bodies]
    ax.legend(handles = handles, loc = "upper left")
    ani = FuncAnimation(fig, update, frames = len(time_values), interval =10)
    
    plt.show()

if __name__ == "__main__":
    main()