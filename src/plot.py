import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import Axes
from matplotlib.lines import Line2D

def draw_core(ax:Axes, R : float, a : float, b : float):
    ax.add_patch(
        patches.Ellipse(
            xy = (R, 0), # xy xy coordinates of ellipse centre.
            width = 2 * a,   # width Total length (diameter) of horizontal axis.
            height = 2 * b, # height Total length (diameter) of vertical axis.
            angle = 0, # angle Rotation in degrees anti-clockwise. 0 by default
            edgecolor = 'black',
            linestyle = 'solid', 
            fill = True,
            facecolor = 'yellow',
            )
    )
    
    return ax

def draw_armour(ax:Axes, R : float, a : float, b : float):
    
    ax.add_patch(
        patches.Polygon(
            np.array([
                [R-a, -0.5*b],
                [R-0.7*a, -b], 
                [R+0.7*a, -b], 
                [R+a, -0.5 * b], 
                [R+a, 0.5*b], 
                [R+0.7*a, b], 
                [R-0.7*a, b],
                [R-a, 0.5*b]
                ]), # xy
            closed=True,
            edgecolor = 'black',
            linestyle = 'solid', 
            fill = True,
            facecolor = 'lightgray',
        )
    )    
    
    return ax

def draw_blanket(ax:Axes, R : float, a_inner : float, b_inner : float, a_outer : float, b_outer : float):
    ax.add_patch(
        patches.Polygon(
            np.array([
                [R-a_outer*1.0, -b_outer * 0.8], 
                [R-a_outer*0.8, -b_outer * 1.0], 
                [R+a_outer*0.7, -b_outer * 1.0], 
                [R+a_outer*0.9, -b_outer * 0.7], 
                [R+a_outer*1.0, -b_outer * 0.5], 
                [R+a_outer*1.0, b_outer * 0.5], 
                [R+a_outer*0.9, b_outer * 0.7], 
                [R+a_outer*0.7, b_outer * 1.0], 
                [R-a_outer*0.8, b_outer * 1.0], 
                [R-a_outer*1.0, b_outer * 0.8]
            ]),
            closed=True,
            edgecolor = 'black',
            linestyle = 'solid', 
            fill = True,
            facecolor = 'blue',
        )
    )    
    
    ax.add_patch(
        patches.Polygon(
            np.array([
                [R-a_inner*1.0, -b_inner * 0.8], 
                [R-a_inner*0.8, -b_inner * 1.0], 
                [R+a_inner*0.7, -b_inner * 1.0], 
                [R+a_inner*0.9, -b_inner * 0.7], 
                [R+a_inner*1.0, -b_inner * 0.5], 
                [R+a_inner*1.0, b_inner * 0.5], 
                [R+a_inner*0.9, b_inner * 0.7], 
                [R+a_inner*0.7, b_inner * 1.0], 
                [R-a_inner*0.8, b_inner * 1.0], 
                [R-a_inner*1.0, b_inner * 0.8]
                ]),
            closed=True,
            edgecolor = 'none',
            linestyle = 'solid', 
            fill = True,
            facecolor = 'white',
        )
    )    
    
    return ax

def draw_coil(ax:Axes,R : float, a_inner : float, b_inner : float, a_outer : float, b_outer : float):
       
    ax.add_patch(
        patches.Polygon(
            np.array([
                [R-a_outer*1.0, -b_outer * 0.8], 
                [R-a_outer*0.8, -b_outer * 1.0], 
                [R+a_outer*0.7, -b_outer * 1.0], 
                [R+a_outer*0.9, -b_outer * 0.7], 
                [R+a_outer*1.0, -b_outer * 0.5], 
                [R+a_outer*1.0, b_outer * 0.5], 
                [R+a_outer*0.9, b_outer * 0.7], 
                [R+a_outer*0.7, b_outer * 1.0], 
                [R-a_outer*0.8, b_outer * 1.0], 
                [R-a_outer*1.0, b_outer * 0.8]
            ]),
            closed=True,
            edgecolor = 'black',
            linestyle = 'solid', 
            fill = True,
            facecolor = 'red',
        )
    )    
    
    ax.add_patch(
        patches.Polygon(
            np.array([
                [R-a_inner*1.0, -b_inner * 0.8], 
                [R-a_inner*0.8, -b_inner * 1.0], 
                [R+a_inner*0.7, -b_inner * 1.0], 
                [R+a_inner*0.9, -b_inner * 0.7], 
                [R+a_inner*1.0, -b_inner * 0.5], 
                [R+a_inner*1.0, b_inner * 0.5], 
                [R+a_inner*0.9, b_inner * 0.7], 
                [R+a_inner*0.7, b_inner * 1.0], 
                [R-a_inner*0.8, b_inner * 1.0], 
                [R-a_inner*1.0, b_inner * 0.8]
                ]),
            closed=True,
            edgecolor = 'none',
            linestyle = 'solid', 
            fill = True,
            facecolor = 'white',
        )
    )    
    
    return ax

def plot_design_poloidal(
    R:float,
    a:float,
    b:float,
    d_armour:float,
    d_blanket:float,
    d_coil:float,
    x_min:float = 0,
    x_max:float = 10,
    y_min:float = -8,
    y_max:float = 8,
    ):
    
    a_armour = a + d_armour
    b_armour = b + d_armour

    a_blanket = a_armour + 0.1
    b_blanket = b_armour + 0.1

    a_coil = a_blanket + d_blanket + 0.1
    b_coil = b_blanket + d_blanket + 0.1

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot()

    ax = draw_coil(ax, R, a_coil, b_coil, a_coil + d_coil, b_coil + d_coil)
    ax = draw_blanket(ax, R, a_blanket, b_blanket, a_blanket + d_blanket, b_blanket + d_blanket)
    ax = draw_armour(ax, R, a_armour, b_armour)
    ax = draw_core(ax, R, a, b)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('R[m]')
    ax.set_ylabel('Z[m]')
    
    legend_elements = [
        Line2D([0],[0], color = 'r', lw = 4, label = "TFC"),
        Line2D([0],[0], color = 'b', lw = 4, label = "Blanekt"),
        Line2D([0],[0], color = 'gray', lw = 4, label = "Armour"),
        Line2D([0],[0], color = 'y', lw = 4, label = "Core"),
        ]
    ax.legend(handles = legend_elements, loc ="upper right")
    return ax