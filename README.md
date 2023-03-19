Lattice Boltzmann Method for fluid simulation
=============================================

This code implements the Lattice Boltzmann Method (LBM) for fluid simulation. The code is organized into several functions that are described below.

get_lattice_constants
---------------------

This function returns the lattice constants that are used in the LBM.

### Parameters

None

### Returns

-   c: np.ndarray of shape (q, 2), where q is the number of lattice velocities. The array contains the lattice velocities.
-   t: np.ndarray of shape (q,), where q is the number of lattice weights. The array contains the lattice weights.
-   noslip: List of length q. It contains the indices of the opposite lattice velocities.
-   i1: List of indices for the unknown velocities on the right wall.
-   i2: List of indices for the unknown velocities in the vertical middle.
-   i3: List of indices for the unknown velocities on the left wall.

mps_to_lu
---------

This function converts a flow velocity from meters per second to lattice units.

### Parameters

-   flow_speed: float. Flow velocity in meters per second.
-   dx: float. Lattice spacing in meters.

### Returns

-   flow_speed_lu: float. Flow velocity in lattice units.

compute_density
---------------

This is a helper function to compute the density from the distribution function.

### Parameters

-   fin: np.ndarray of shape (q, nx, ny), where q is the number of lattice velocities, and nx, ny are the number of lattice nodes in the x and y directions respectively. The array contains the distribution functions.

### Returns

-   rho: np.ndarray of shape (nx, ny). The array contains the computed density at each node.

equilibrium_distribution
------------------------

This function computes the equilibrium distribution function.

### Parameters

-   rho: np.ndarray of shape (nx, ny). The density.
-   u: np.ndarray of shape (2, nx, ny). The velocity.
-   c: np.ndarray of shape (q, 2), where q is the number of lattice velocities. The array contains the lattice velocities.
-   t: np.ndarray of shape (q,), where q is the number of lattice weights. The array contains the lattice weights.
### Returns

-   feq: np.ndarray of shape (q, nx, ny), where q is the number of lattice velocities. The array contains the equilibrium distribution functions.

create_dir_if_not_exists
------------------------

This function checks if a directory exists at the specified path and creates it if it doesn't.

### Parameters

-   dir_path : str. Path of the directory to check/create.

### Returns

None

create_obstacle
---------------

This function creates an obstacle in the form of a numpy ndarray based on the specified shape.

### Parameters

-   shape : str. String representing the shape of the obstacle. Possible values: "circle", "rectangle", "square".
-   nx : int. Grid size in the x-direction.
-   ny : int. Grid size in the y-direction.
-   cx : int. x-coordinate of the center of the obstacle.
-   cy : int. y-coordinate of the center of the obstacle.
-   r : int. Radius of the obstacle (for "circle" shape).
-   l : int (optional). Length of the obstacle (for "rectangle" shape).
-   w : int (optional). Width of the obstacle (for "rectangle" and "square" shapes).

### Returns

-   numpy.ndarray. Numpy array representing the obstacle.

load_mask
---------

This function loads a binary mask from a PNG file and scales it by the given factor.

### Parameters

-   filename : str. Name of the PNG file to load.
-   scale : float, optional. Scaling factor for the mask (default= 1).

Returns mask : numpy.ndarray. Binary mask of shape (height, width).

def load_mask(filename, scale=1): img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) if scale != 1: img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_NEAREST) mask = (img > 0).astype(np.uint8) return mask

In this function, we first read the PNG file using OpenCV's imread() function with the flag cv2.IMREAD_GRAYSCALE to load the image as a grayscale image. Then, if a scaling factor is given, we resize the image using OpenCV's resize() function with the interpolation method set to cv2.INTER_NEAREST. The cv2.INTER_NEAREST method is used to preserve the binary nature of the mask after scaling.

Finally, we convert the image to a binary mask by checking if the pixel values are greater than 0 using the > operator and then converting the resulting Boolean array to a uint8 array using astype(np.uint8). This gives us a binary mask where the object of interest is represented by white pixels (pixel value of 1) and the background is represented by black pixels (pixel value of 0).