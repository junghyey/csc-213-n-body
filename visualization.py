import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import pandas as pd
from matplotlib import cm


def main():
    # Load n-body simulation data from a CSV file
    output = pd.read_csv("./output.csv")
    output_groupped = output.groupby("Time")

    # Initialize the figure
    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(projection = '3d')
    time_values = sorted(output["Time"].unique())
    unique_bodies = sorted(output["Body"].unique())
    cmap = plt.cm.hsv
    body_colors = {body: cmap(i/len(unique_bodies))[:3] for i,body in enumerate(unique_bodies)}

    # Determine size range for plotting based on mass
    max_size = max(output["Mass"])
    min_size = min(output["Mass"])

  # Set axis limits and labels
    ax.set_xlim(output['Px'].min(), output['Px'].max())
    ax.set_ylim(output['Py'].min(), output['Py'].max())
    ax.set_zlim(output['Pz'].min(), output['Pz'].max())
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_zlabel("Z-coordinate")

    '''
    Update visualization function for each frame of the animation.
    '''
    def update(frame):
        ax.clear() # Clear previous frame's data

        # Get the snapshot of data for the current frame
        snapshot = output_groupped.get_group(time_values[frame])

        # Reset axis limits and labels
        ax.set_title(f"Time: {time_values[frame]:.1f}s")
        ax.set_xlim(output['Px'].min(), output['Px'].max())
        ax.set_ylim(output['Py'].min(), output['Py'].max())
        ax.set_zlim(output['Pz'].min(), output['Pz'].max())
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_zlabel("Z-coordinate")

        # Plot each body's position
        for body in unique_bodies:
            body_data = snapshot[snapshot["Body"] == body]
            ax.scatter(body_data["Px"], 
                       body_data["Py"], 
                       body_data["Pz"], 
                       c =[body_colors[body]], 
                       s = 50+ (200-50) * (body_data["Mass"] - min_size)/(max_size-min_size), 
                       marker = "*")

        if len(unique_bodies) <= 25: 
          # Add the legend
          handles = [plt.Line2D ([], [], color = body_colors[body], 
                                marker = "o", 
                                label = f"Body {body}") for body in unique_bodies]
          ax.legend(handles = handles, 
                    loc = "center left", 
                    bbox_to_anchor = (1.1, 0.5), 
                    title= "Bodies", 
                    ncol = 2)

    # Create an animation using the update function
    ani = FuncAnimation(fig, update, frames = len(time_values), interval =50)

    # Show the animation
    plt.show()

    # Save the animation as a GIF
    #ani.save("visualization.gif", writer=PillowWriter(fps=30))
    
if __name__ == "__main__":
    main()