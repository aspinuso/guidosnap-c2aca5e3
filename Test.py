import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#Load a file:
index = netCDF4.Dataset("ffmcfile.nc", "r")
#Choose the index, for now FFMC or ISI
type_of_index = 'ffmc'

#Finding dimensions of one of the indeces:
print(index.variables[type_of_index].shape)

#Make and save the pictures of the first (in this case) 12 days
i=1
while i<12:
    plt.figure()
    # Here 'useful' values for vmin and vmax for in this case ffmc have been chosen. 80 to 90 gives nice pictures, for now
    #80 to 100/105, would give a more realistic 'view of danger'. For ISI use
    plt.imshow(index[type_of_index][i,::-1,:], cmap='coolwarm', vmin=80, vmax=90)
    #making the bar on the right side

    plt.colorbar(extend='both')
    #Displaying the title
    plt.title(type_of_index + " " + str(i) + " mei")
    plt.savefig(type_of_index + " " + str(i) + " mei.png")
    i+=1


