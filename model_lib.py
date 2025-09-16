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
def mps_to_lu(flow_speed: float, dx: float, dt: float = 1.0) -> float:
    """
    Converts a flow velocity from meters per second to lattice units.
    
    In the Lattice Boltzmann Method, the conversion from physical units to lattice units
    is given by: u_LU = u_physical * (dt / dx), where dt is the time step in seconds
    and dx is the lattice spacing in meters.
    
    Args:
    - flow_speed: float. Flow velocity in meters per second.
    - dx: float. Lattice spacing in meters.
    - dt: float. Time step in seconds (default=1.0 for unit time step).
    
    Returns:
    - flow_speed_lu: float. Flow velocity in lattice units.
    """
    # Correct conversion: lattice velocity = physical velocity * (dt / dx)
    flow_speed_lu = flow_speed * (dt / dx)
    
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


def equilibrium_distribution(q: int, nx: int, ny: int, rho: np.ndarray, u: np.ndarray, c: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes the equilibrium distribution function for the Lattice Boltzmann Method.
    
    This function implements the Maxwell-Boltzmann equilibrium distribution expanded
    to second order in the Mach number for the D2Q9 lattice.
    
    Args:
    - q: int. Number of lattice velocities (should be 9 for D2Q9).
    - nx: int. Number of grid points in x direction.
    - ny: int. Number of grid points in y direction.
    - rho: np.ndarray of shape (nx, ny). The density field.
    - u: np.ndarray of shape (2, nx, ny). The velocity field.
    - c: np.ndarray of shape (q, 2). Lattice velocity vectors.
    - t: np.ndarray of shape (q,). Lattice weights.
    
    Returns:
    - feq: np.ndarray of shape (q, nx, ny). Equilibrium distribution functions.
    """
    # Validate inputs
    assert u.shape[0] == 2, "Velocity field must have shape (2, nx, ny)"
    assert c.shape[0] == q, "Lattice velocities must have shape (q, 2)"
    assert t.shape[0] == q, "Lattice weights must have shape (q,)"
    
    # Compute dot product of lattice velocities with macroscopic velocity
    cu = 3.0 * np.dot(c, u.transpose(1, 0, 2))
    
    # Compute velocity magnitude squared
    usqr = 3.0/2.0 * (u[0]**2 + u[1]**2)
    
    # Initialize equilibrium distribution
    feq = np.zeros((q, nx, ny))
    
    # Compute equilibrium distribution for each lattice direction
    for i in range(q): 
        feq[i, :, :] = rho * t[i] * (1.0 + cu[i] + 0.5*cu[i]**2 - usqr)
    
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

def setup_cylinder_obstacle_and_perturbation(q: int, rho: float, nx: int, ny: int, cx: float, cy: float, r: float, uLB: float, ly: float, c: np.ndarray, t: np.ndarray, epsilon: float = 1e-4) -> tuple:
    """
    Sets up the cylindrical obstacle and velocity inlet with perturbation for the Lattice Boltzmann Method.
    
    Args:
    - q: int. Number of lattice velocities (should be 9 for D2Q9).
    - rho: float. Initial density value.
    - nx: int. Number of lattice nodes in the x direction.
    - ny: int. Number of lattice nodes in the y direction.
    - cx: float. x-coordinate of the center of the cylinder.
    - cy: float. y-coordinate of the center of the cylinder.
    - r: float. Radius of the cylinder.
    - uLB: float. Magnitude of the inlet velocity in lattice units.
    - ly: float. Height of the domain in lattice units.
    - c: np.ndarray of shape (q, 2). Lattice velocity vectors.
    - t: np.ndarray of shape (q,). Lattice weights.
    - epsilon: float. Magnitude of the perturbation in the velocity inlet.
    
    Returns:
    - tuple: (fin, vel, obstacle) where:
        - fin: Distribution functions of shape (q, nx, ny)
        - vel: Velocity field of shape (2, nx, ny)  
        - obstacle: Boolean obstacle mask of shape (nx, ny)
    """
    # Set up rectangular obstacle (can be changed to circle if needed)
    obstacle = create_obstacle("rectangle", nx, ny, int(cx), int(cy), int(r), int(r), int(r))
    
    # Set up velocity inlet with perturbation
    vel = np.fromfunction(lambda d, x, y: (1-d)*uLB*(1.0 + epsilon*np.sin(y/ly*2*np.pi)), (2, nx, ny))
    
    # Create uniform density field
    rho_field = rho * np.ones((nx, ny))
    
    # Compute the equilibrium distribution function
    feq = equilibrium_distribution(q, nx, ny, rho_field, vel, c, t)
    fin = feq.copy()
    
    return fin, vel, obstacle

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


def compute_fluid_flow(q: int, nx: int, ny: int, fin: np.ndarray, vel: np.ndarray, obstacle: np.ndarray, omega: float, t: np.ndarray, c: np.ndarray, i1: np.ndarray, i2: np.ndarray, i3: np.ndarray, noslip: np.ndarray) -> tuple:
    """
    Computes fluid flow using Lattice Boltzmann Method.
    
    Parameters:
    -----------
    q : int
        Number of lattice velocities (should be 9 for D2Q9).
    nx : int
        Number of lattice nodes in x direction.
    ny : int
        Number of lattice nodes in y direction.
    fin : numpy.ndarray
        Distribution functions of shape (q, nx, ny).
    vel : numpy.ndarray
        Velocity field of shape (2, nx, ny).
    obstacle : numpy.ndarray
        Boolean array marking obstacle locations of shape (nx, ny).
    omega : float
        Relaxation parameter (0 < omega < 2).
    t : numpy.ndarray
        Lattice weights of shape (q,).
    c : numpy.ndarray
        Lattice velocities of shape (q, 2).
    i1 : numpy.ndarray
        Indices for unknown velocities on the right wall.
    i2 : numpy.ndarray
        Indices for unknown velocities in the vertical middle.
    i3 : numpy.ndarray
        Indices for unknown velocities on the left wall.
    noslip : numpy.ndarray
        Indices for bounce-back boundary condition.
    
    Returns:
    --------
    tuple
        (fin, u, rho, feq, fout, pressure) - Updated distribution functions, 
        velocity field, density field, equilibrium distributions, 
        post-collision distributions, and pressure field.
    """
    
    # Input validation
    assert 0.0 < omega < 2.0, f"Relaxation parameter omega must be between 0 and 2, got {omega}"
    assert fin.shape == (q, nx, ny), f"Distribution function shape mismatch: expected ({q}, {nx}, {ny}), got {fin.shape}"
    assert vel.shape == (2, nx, ny), f"Velocity field shape mismatch: expected (2, {nx}, {ny}), got {vel.shape}"
    assert obstacle.shape == (nx, ny), f"Obstacle array shape mismatch: expected ({nx}, {ny}), got {obstacle.shape}"
    
    # Set reflective boundary conditions at the top and bottom boundaries.
    fin[:, 0, :] = fin[:, 1, :]     # top boundary (y=0)
    fin[:, -1, :] = fin[:, -2, :]   # bottom boundary (y=ny-1)
    
    # Right wall: outflow condition
    fin[i1, -1, :] = fin[i1, -2, :] 
    
    # Calculate macroscopic density and velocity
    rho = compute_density(fin)  
    u = np.dot(c.transpose(), fin.transpose((1, 0, 2))) / rho

    # Left wall: apply inlet velocity boundary condition
    u[:, 0, :] = vel[:, 0, :]
    
    # Left wall: compute density from known populations using Zou/He boundary condition
    # Avoid division by zero
    denominator = 1.0 - u[0, 0, :]
    safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    rho[0, :] = (1.0 / safe_denominator) * (compute_density(fin[i2, 0, :]) + 2.0 * compute_density(fin[i1, 0, :]))

    # Compute equilibrium distribution
    feq = equilibrium_distribution(q, nx, ny, rho, u, c, t) 
    
    # Left wall: Zou/He boundary condition for unknown distributions
    fin[i3, 0, :] = fin[i1, 0, :] + feq[i3, 0, :] - fin[i1, 0, :]
    
    # Collision step (BGK approximation)
    fout = fin - omega * (fin - feq)  
    
    # Apply bounce-back boundary condition at obstacles
    for i in range(q): 
        fout[i, obstacle] = fin[noslip[i], obstacle]
    
    # Streaming step
    for i in range(q): 
        fin[i, :, :] = np.roll(np.roll(fout[i, :, :], c[i, 0], axis=0), c[i, 1], axis=1)
    
    # Calculate pressure using proper equation of state for LBM
    # In lattice units, pressure = density * cs^2, where cs^2 = 1/3 for D2Q9
    rho = compute_density(fin)
    cs_squared = 1.0/3.0  # Speed of sound squared in lattice units
    pressure = rho * cs_squared
    
    return fin, u, rho, feq, fout, pressure

def validate_lattice_boltzmann_properties(fin: np.ndarray, c: np.ndarray, t: np.ndarray, tolerance: float = 1e-10) -> dict:
    """
    Validates key mathematical properties of the Lattice Boltzmann Method.
    
    Parameters:
    -----------
    fin : np.ndarray
        Distribution functions of shape (q, nx, ny).
    c : np.ndarray
        Lattice velocities of shape (q, 2).
    t : np.ndarray
        Lattice weights of shape (q,).
    tolerance : float
        Numerical tolerance for validation checks.
    
    Returns:
    --------
    dict
        Dictionary containing validation results and computed properties.
    """
    results = {}
    
    # Check if weights sum to 1
    weight_sum = np.sum(t)
    results['weights_sum_to_one'] = np.abs(weight_sum - 1.0) < tolerance
    results['weight_sum'] = weight_sum
    
    # Check mass conservation (sum of distributions should equal density)
    rho = compute_density(fin)
    mass_conservation = np.allclose(np.sum(fin, axis=0), rho, rtol=tolerance)
    results['mass_conservation'] = mass_conservation
    
    # Check momentum conservation
    momentum = np.zeros((2, fin.shape[1], fin.shape[2]))
    for i in range(fin.shape[0]):
        momentum[0] += fin[i] * c[i, 0]
        momentum[1] += fin[i] * c[i, 1]
    
    results['momentum_field'] = momentum
    results['max_momentum_x'] = np.max(np.abs(momentum[0]))
    results['max_momentum_y'] = np.max(np.abs(momentum[1]))
    
    # Validate lattice structure (D2Q9)
    expected_velocities = np.array([
        [0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], 
        [-1, 1], [1, 0], [1, -1], [1, 1]
    ])
    lattice_structure_valid = np.allclose(c, expected_velocities)
    results['lattice_structure_valid'] = lattice_structure_valid
    
    return results


def check_stability_conditions(u: np.ndarray, omega: float, Ma_max: float = 0.1) -> dict:
    """
    Checks stability conditions for the Lattice Boltzmann simulation.
    
    Parameters:
    -----------
    u : np.ndarray
        Velocity field of shape (2, nx, ny).
    omega : float
        Relaxation parameter.
    Ma_max : float
        Maximum allowed Mach number for stability.
    
    Returns:
    --------
    dict
        Dictionary containing stability analysis results.
    """
    results = {}
    
    # Check Mach number (velocity should be much less than sound speed)
    cs = 1.0 / np.sqrt(3.0)  # Sound speed in lattice units for D2Q9
    velocity_magnitude = np.sqrt(u[0]**2 + u[1]**2)
    mach_number = velocity_magnitude / cs
    max_mach = np.max(mach_number)
    
    results['max_mach_number'] = max_mach
    results['mach_stable'] = max_mach < Ma_max
    results['sound_speed'] = cs
    
    # Check relaxation parameter stability (should be between 0 and 2)
    results['omega'] = omega
    results['omega_stable'] = 0.0 < omega < 2.0
    
    # Compute Reynolds number based on maximum velocity
    if np.max(velocity_magnitude) > 0:
        nu = (1.0/omega - 0.5) / 3.0  # Kinematic viscosity in lattice units
        # Estimate characteristic length (could be improved with actual obstacle size)
        L_char = np.sqrt(u.shape[1] * u.shape[2]) / 10.0  # Rough estimate
        Re_estimate = np.max(velocity_magnitude) * L_char / nu
        results['estimated_reynolds'] = Re_estimate
        results['kinematic_viscosity'] = nu
    
    return results
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

