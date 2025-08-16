import numpy as np
import matplotlib.pyplot as plt

zeta_values = np.linspace(0, 1, 101) # dividing zeta span that is from 0 to 1 into 101 equal spans
eta_values = np.linspace(0, 1, 81) # dividing eta span that is from 0 to 1 into 81 equal spans
grid_x = []
grid_y = []

for eta in eta_values: # for the value of each eta, zeta is iterated for given number of times to obtain a set of zeta that corresponds to that specific eta
    dummy_x = []  #to fill x coordinates temporarily
    dummy_y = []  #to fill y coordinates temporarily
    for zeta in zeta_values:
        dummy_x.append((1 + 19*eta)*np.cos(2*np.pi*zeta)) 
        dummy_y.append((0.5*(1 + 39*eta))*np.sin(2*np.pi*zeta))
    grid_x.append(dummy_x)
    grid_y.append(dummy_y)

grid_x = np.array(grid_x)#creating arrays out of points to form array out of it
grid_y = np.array(grid_y)

plt.figure(figsize=(10, 10)) # specifying the size of the plot
plt.plot(grid_x, grid_y,'k', linewidth=0.5)  # Plot eta lines
plt.plot(grid_x.T, grid_y.T,'k', linewidth=0.5)  # Plot zeta lines

theta = np.linspace(0, 2*np.pi, 200)
x_inner = np.cos(theta) #x coordinate of ellipse
y_inner = 0.5*np.sin(theta) #y coordinate of ellipse
x_outer = 20*np.cos(theta) #x coordinate of circle
y_outer = 20*np.sin(theta) #y coordinate of circle

plt.plot(x_inner, y_inner, 'r')#plotting Ellipse
plt.plot(x_outer, y_outer, 'b')#plotting Circle

# Set equal aspect ratio
plt.axis("equal")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Transfinite Interpolation (TFI) Grid Generation")
plt.show()
