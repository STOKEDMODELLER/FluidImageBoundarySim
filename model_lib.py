from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def get_lattice_constants() -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[int]]:
    """
    This function returns the lattice constants that are used in the Lattice Boltzmann Method.
    
    Returns:
    - c: np.ndarray of shape (q, 2), where q is the number of lattice velocities.
         The array contains the lattice velocities.
    - t: np.ndarray of shape (q,), where q is the number of lattice weights.
         The array contains the lattice weights.
    - noslip: List of length q. It contains the indices of the opposite lattice velocities.
    - i1: List of indices for the unknown velocities on the right wall.
    - i2: List of indices for the unknown velocities in the vertical middle.
    - i3: List of indices for the unknown velocities on the left wall.
    """
    c = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
    q = c.shape[0]
    
    t = 1./36. * np.ones(q)  # Lattice weights.
    norm_ci = np.array([np.linalg.norm(ci) for ci in c])
    t[norm_ci < 1.1] = 1./9.
    t[0] = 4./9.
    
    noslip = [np.where((c == -c[i]).all(axis=1))[0][0] for i in range(q)]
    
    i1 = np.where(c[:, 0] < 0)[0].tolist()
    i2 = np.where(c[:, 0] == 0)[0].tolist()
    i3 = np.where(c[:, 0] > 0)[0].tolist()
    
    return c, t, noslip, i1, i2, i3
def mps_to_lu(flow_speed: float, dx: float) -> float:
    """
    Converts a flow velocity from meters per second to lattice units.
    
    Args:
    - flow_speed: float. Flow velocity in meters per second.
    - dx: float. Lattice spacing in meters.
    
    Returns:
    - flow_speed_lu: float. Flow velocity in lattice units.
    """
    # Compute the flow velocity in lattice units.
    flow_speed_lu = flow_speed * (dx / 1.0)
    
    return flow_speed_lu

def compute_density(fin: np.ndarray) -> np.ndarray:
    """
    Helper function to compute the density from the distribution function.
    
    Args:
    - fin: np.ndarray of shape (q, nx, ny), where q is the number of lattice velocities,
           and nx, ny are the number of lattice nodes in the x and y directions respectively.
           The array contains the distribution functions.
    
    Returns:
    - rho: np.ndarray of shape (nx, ny). The array contains the computed density at each node.
    """
    return np.sum(fin, axis=0)


def equilibrium_distribution(q: float,nx: float,ny: float,rho: np.ndarray, u: np.ndarray, c: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes the equilibrium distribution function.
    
    Args:
    - rho: np.ndarray of shape (nx, ny). The density.
    - u: np.ndarray of shape (2, nx, ny). The velocity.
    - c: np.ndarray of shape (q, 2), where q is the number of lattice velocities.
         The array contains the lattice velocities.
    - t: np.ndarray of shape (q,), where q is the number of lattice weights.
         The array contains the lattice weights.
    
    Returns:
    - feq: np.ndarray of shape (q, nx, ny), where q is the number of lattice velocities.
           The array contains the equilibrium distribution functions.
    """
    cu   = 3.0 * np.dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = np.zeros((q,nx,ny))
    for i in range(q): 
        feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq
import os

def create_dir_if_not_exists(dir_path: str):
    """
    Checks if a directory exists at the specified path and creates it if it doesn't.
    
    Parameters:
    -----------
    dir_path : str
        Path of the directory to check/create.
    """
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_obstacle(shape: str, nx: int, ny: int, cx: int, cy: int, r: int, l: int = 0, w: int = 0) -> np.ndarray:
    """
    Creates an obstacle in the form of a numpy ndarray based on the specified shape.
    
    Parameters:
    -----------
    shape : str
        String representing the shape of the obstacle.
        Possible values: "circle", "rectangle", "square".
    nx : int
        Grid size in the x-direction.
    ny : int
        Grid size in the y-direction.
    cx : int
        x-coordinate of the center of the obstacle.
    cy : int
        y-coordinate of the center of the obstacle.
    r : int
        Radius of the obstacle (for "circle" shape).
    l : int
        Length of the obstacle (for "rectangle" shape).
    w : int
        Width of the obstacle (for "rectangle" and "square" shapes).
    
    Returns:
    --------
    numpy.ndarray
        Numpy array representing the obstacle.
    """
    
    if shape == "circle":
        obstacle = np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))
    elif shape == "rectangle":
        obstacle = np.fromfunction(lambda x,y: (x>cx-l/2) & (x<cx+l/2) & (y>cy-w/2) & (y<cy+w/2), (nx,ny))
    elif shape == "square":
        obstacle = np.fromfunction(lambda x,y: (x>cx-l/2) & (x<cx+l/2) & (y>cy-w/2) & (y<cy+w/2), (nx,ny))
    else:
        raise ValueError(f"Invalid obstacle shape: {shape}")
    
    return obstacle


from PIL import Image

def load_mask(filename: str, scale: float = 1.0) -> np.ndarray:
    """
    Loads a binary mask from a PNG file and scales it by the given factor.
    
    Parameters:
    -----------
    filename : str
        Name of the PNG file to load.
    scale : float, optional
        Scaling factor for the mask (default is 1.0).
    
    Returns:
    --------
    numpy.ndarray
        Numpy array representing the binary mask.
    """
    
    # Load the PNG file using PIL.
    im = Image.open(filename).convert("L")
    
    # Scale the image using PIL.
    size = tuple(int(dim * scale) for dim in im.size)
    im = im.resize(size)
    
    # Convert the PIL image to a numpy array.
    mask = np.array(im)
    
    # Convert the values to binary (0 or 1).
    mask = (mask > 0).astype(int)
    
    return np.logical_not(mask)


from scipy.ndimage import rotate

def check_neighbors(matrix: np.ndarray) -> np.ndarray:
    """
    Checks if each False cell in a binary matrix is surrounded by three True cells
    (including diagonal neighbors), and sets it to True if it is.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix to check.
    
    Returns:
    --------
    numpy.ndarray
        Binary matrix with updated values.
    """
    
    # Copy the input matrix
    output = np.copy(matrix)
    
    # Find the cells that meet the condition (i.e., four adjacent True cells)
    condition = (~matrix[1:-1, 1:-1] &
                 matrix[1:-1, :-2] & matrix[1:-1, 2:] &
                 matrix[:-2, 1:-1] & matrix[2:, 1:-1])
    
    # Set the corresponding cells in the output matrix to True
    output[1:-1, 1:-1][condition] = True
    
    return output
from scipy.ndimage import binary_erosion, generate_binary_structure

def smooth_corners(obstacle_mask: np.ndarray, erosion_size: int = 2) -> np.ndarray:
    """
    Smooths the corners of an obstacle mask by applying an erosion operation.
    Assumes that True values represent obstacles.
    
    Parameters:
    -----------
    obstacle_mask : numpy.ndarray
        Binary mask representing obstacles.
    erosion_size : int, optional
        Size of the erosion structuring element (default is 2).
    
    Returns:
    --------
    numpy.ndarray
        Binary mask with smoothed corners.
    """
    
    # Create a structuring element for erosion
    selem = generate_binary_structure(2, 2)
    selem[erosion_size-1:, erosion_size-1:] = False
    
    # Apply erosion to the mask
    smoothed_mask = binary_erosion(obstacle_mask, selem)
    
    return smoothed_mask
def rotate_obstacle_mask(obstacle_mask: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates the True section of an obstacle mask by a given angle in degrees.
    Assumes that True values represent obstacles.
    
    Parameters:
    -----------
    obstacle_mask : numpy.ndarray
        Binary mask representing obstacles.
    angle : float
        Angle of rotation in degrees (counterclockwise).
    
    Returns:
    --------
    numpy.ndarray
        Rotated binary mask representing obstacles.
    """
    
    # Find the indices of the True values in the mask
    indices = np.argwhere(obstacle_mask)
    
    # Calculate the centroid of the True section
    centroid = np.mean(indices, axis=0)
    
    # Translate the indices to the centroid
    indices_centered = indices - centroid
    
    # Rotate the indices by the given angle
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                [np.sin(radians), np.cos(radians)]])
    indices_rotated = np.dot(indices_centered, rotation_matrix).astype(int)
    
    # Translate the rotated indices back to their original position
    indices_uncentered = indices_rotated + centroid
    
    # Create a new mask with the same shape as the input mask
    rotated_mask = np.zeros_like(obstacle_mask)
    
    # Set the values at the rotated indices to True
    indices_uncentered = indices_uncentered.astype(int)
    indices_uncentered[:, 0] = np.clip(indices_uncentered[:, 0], 0, rotated_mask.shape[0]-1)
    indices_uncentered[:, 1] = np.clip(indices_uncentered[:, 1], 0, rotated_mask.shape[1]-1)
    rotated_mask[indices_uncentered[:, 0], indices_uncentered[:, 1]] = True
    
    return rotated_mask


def plot_velocity_pressure(u, pressure, time, directory,fig, ax,cmm):
    """
    Plot velocity magnitude and pressure fields and save them as png files.

    Parameters:
    u (numpy.ndarray): velocity field.
    pressure (numpy.ndarray): pressure field.
    time (float): current simulation time.
    directory (str): path to directory where the files will be saved.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Create subplots and adjust spacing
    fig.subplots_adjust(hspace=0.01)

    # Plot velocity magnitude
    ax1.imshow(np.sqrt(u[0]**2+u[1]**2).transpose(), cmap=cmm, vmax=0.08, vmin=0)
    ax1.set_title('Velocity Magnitude')
    ax1.axis('off')

    # Plot pressure
    ax2.imshow(pressure.transpose(), cmap=cmm, vmax=0.006, vmin=0.001)
    ax2.set_title('Pressure')
    ax2.axis('off')

    # Save figure
    fig.tight_layout()
    fig.savefig(f"{directory}vel_pressure.{time/100:.4f}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def setup_cylinder_obstacle_and_perturbation(q: float, rho: float,nx: int, ny: int, cx: float, cy: float, r: float, uLB: float, ly: float, c: np.ndarray, t: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Sets up the cylindrical obstacle and velocity inlet with perturbation for the Lattice Boltzmann Method.
    
    Args:
    - nx: int. Number of lattice nodes in the x direction.
    - ny: int. Number of lattice nodes in the y direction.
    - cx: float. x-coordinate of the center of the cylinder.
    - cy: float. y-coordinate of the center of the cylinder.
    - r: float. Radius of the cylinder.
    - uLB: float. Magnitude of the inlet velocity.
    - ly: float. Height of the domain in lattice units.
    - c: np.ndarray of shape (q, 2), where q is the number of lattice velocities.
         The array contains the lattice velocities.
    - t: np.ndarray of shape (q,), where q is the number of lattice weights.
         The array contains the lattice weights.
    - epsilon: float. Magnitude of the perturbation in the velocity inlet.
    
    Returns:
    - fin: np.ndarray of shape (q, nx, ny), where q is the number of lattice velocities,
           and nx, ny are the number of lattice nodes in the x and y directions respectively.
           The array contains the distribution functions.
    """
    # Set up cylindrical obstacle.
    # obstacle = create_obstacle("circle", nx, ny, cx, cy, r)
    obstacle = create_obstacle("rectangle", nx, ny, cx, cy, r, r, r)
    
    # Set up velocity inlet with perturbation.
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    vel = np.fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*np.sin(y/ly*2*np.pi)),(2,nx,ny))
    # Compute the equilibrium distribution function.
    feq = equilibrium_distribution(q,nx,ny,rho, vel,(c), t)
    fin = feq.copy()
    sumpop = sum_population(fin)
    return fin,vel,obstacle,sumpop

def sum_population(fin: np.ndarray) -> np.ndarray:
    """
    Computes the sum of elements of the input numpy array along the axis 0.
    
    Parameters:
    -----------
    fin : numpy.ndarray
        Input numpy array
    
    Returns:
    --------
    numpy.ndarray
        Sum of elements of the input array along the axis 0.
    """
    return np.sum(fin, axis=0)


def compute_fluid_flow(sumpop,q: float,nx: int, ny: int,fin: np.ndarray, vel: np.ndarray, obstacle: np.ndarray, omega: float, t: float, c: np.ndarray, i1: np.ndarray, i2: np.ndarray, i3: np.ndarray, noslip: np.ndarray) -> np.ndarray:
    """
    Computes fluid flow using Lattice Boltzmann Method.
    
    Parameters:
    -----------
    fin : numpy.ndarray
        Input numpy array
    vel : numpy.ndarray
        Numpy array containing velocity components.
    obstacle : numpy.ndarray
        Numpy array containing obstacle data.
    omega : float
        Relaxation parameter.
    t : float
        Time
    c : numpy.ndarray
        Numpy array containing the 3D velocity vectors for each of the 19 discrete directions.
    i1 : numpy.ndarray
        Numpy array used to retrieve populations.
    i2 : numpy.ndarray
        Numpy array used to retrieve indices of populations.
    i3 : numpy.ndarray
        Numpy array used to retrieve opposite indices of populations.
    noslip : numpy.ndarray
        Numpy array containing indices for no-slip boundary condition.
    
    Returns:
    --------
    numpy.ndarray
        Computed fluid flow.
    """
    
    # Set reflective boundary conditions at the top and bottom boundaries.
    fin[:,0,:] = fin[:,1,:]  # top boundary
    fin[:,-1,:] = fin[:,-2,:]  # bottom boundary
    
    fin[i1,-1,:] = fin[i1,-2,:] # Right wall: outflow condition.
    rho = sumpop(fin)           # Calculate macroscopic density and velocity.
    u = np.dot(c.transpose(), fin.transpose((1,0,2)))/rho

    u[:,0,:] =vel[:,0,:] # Left wall: compute density from known populations.
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))

    feq = equilibrium_distribution(q,nx,ny,rho, u, (c), t) # Left wall: Zou/He boundary condition.
    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]
    fout = fin - omega * (fin - feq)  # Collision step.
    for i in range(q): fout[i,obstacle] = fin[noslip[i],obstacle]
    for i in range(q): # Streaming step.
        fin[i,:,:] = np.roll(np.roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)
    
    # Calculate pressure.
    rho = compute_density(fin)
    vel_mag = np.sqrt(u[0]**2 + u[1]**2)
    pressure = rho * vel_mag**2
    return fin,u,rho,feq,fout,pressure

import numpy as np

def plot_velocity(velocity, time, directory, fig, ax, colormap, vmin,vmax,dpi=80):
    """
    Plots the velocity field at a given time step and saves the image to a specified directory.

    Args:
        velocity (numpy.ndarray): The velocity field.
        time (int): The time step of the simulation.
        directory (str): The directory where the image should be saved.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        colormap (str, optional): The colormap to use. Defaults to 'jet'.
        dpi (int, optional): The dots per inch for the saved image. Defaults to 80.
    """

    ax.clear()
    img = ax.imshow(np.sqrt(velocity[0] ** 2 + velocity[1] ** 2).T, cmap=colormap,vmin=vmin,vmax=vmax)
    ax.set_title('Velocity field (Lattice units)\nTime step: {}'.format(time))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # fig.colorbar(img, ax=ax, orientation='vertical', label='Velocity Magnitude (Lattice units)')
    plt.tight_layout()

    output_image = f"{directory}vel_pressure.{time/100:.4f}.png"
    plt.savefig(output_image, dpi=dpi)

