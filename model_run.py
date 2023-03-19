import model_lib as ml
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

obstacle1 = ml.load_mask("./bitmap667.png",1).transpose()
obstacle = obstacle1
plt.imshow(obstacle)
# Set the directory where the images are located
directory = './Ln_5/'
ml.create_dir_if_not_exists(directory)

###### Flow definition #########################################################
maxIter = 30000 # Total number of time iterations.
Re      = 300.0  # Reynolds number.
nx = obstacle.shape[0]; ny = obstacle.shape[1]; ly=ny-1.0; q = 9 # Lattice dimensions and populations.
cx = nx/4; cy=ny/2; r=ny/9;          # Coordinates of the cylinder.
uLB     = ml.mps_to_lu(0.05,ly / (ny - 1))                       # Velocity in lattice units.
nulb    = uLB*r/Re; omega = 1.0 / (3.*nulb+0.5); # Relaxation parameter.
rho = 1

c, t, noslip, i1, i2, i3 = ml.get_lattice_constants()

fin, vel, obstacle_1, sumpop = ml.setup_cylinder_obstacle_and_perturbation(q,rho,nx, ny, cx, cy, r, uLB, ly,c,t, epsilon = 1e-4)

sumpop = lambda fin: np.sum(fin, axis=0)

###### Main time loop ##########################################################

fig, ax = plt.subplots(figsize=(16,9))

for time in range(maxIter):
    print(time)
    fin, u, rho, feq, fin_collide, pressure = ml.compute_fluid_flow(sumpop,q,nx, ny,fin, vel, obstacle, omega, t, c, i1, i2, i3, noslip)
    if (time%50==0): 
        ml.plot_velocity(u, time, directory, fig, ax, 'gist_gray',0,0.1, dpi=100)

 
# Set the output video file name
output_file = directory+'output.mp4'

# Set the video frame rate
fps = 15

# Get all the image file names in the directory
img_files = [f for f in os.listdir(directory) if f.endswith('.png')]

# Define a function to extract the timestamp from the file name
def get_timestamp(filename):
    return float(filename.split('.')[1])

# Sort the image file list by the timestamp
img_files = sorted(img_files, key=get_timestamp)

# Read the first image to get the image size
img = cv2.imread(directory + img_files[0])
height, width, channels = img.shape

# Create the video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Loop through all the image files and write them to the video file
for img_file in img_files:
    img = cv2.imread(directory + img_file)
    out.write(img)

# Release the video writer object and print a message
out.release()
print('Video saved as ' + output_file)



