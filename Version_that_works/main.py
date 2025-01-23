"""
Here the matrices are created that give rise to the pictures of the fire indices we get.

"""
#import databases
from datetime import datetime
import matplotlib.pyplot as plt
import netCDF4
from FireWeatherIndex import FWICLASS
import numpy as np
from tqdm import tqdm
from MatrixCalculations import MATRIXCALC

#measure time
startTime = datetime.now()


#starting constant, for ffmc0. 85 means that the whole matrix start at 85 for the ffmc, this is about the value, where
#it has been dry for 2 or 3 days and not too hot.
#Starting values:
ffmc0 = 85.0


#loading files
temp = netCDF4.Dataset("tg_2024050112-14H.nc", "r")
rhum = netCDF4.Dataset("hu_2024050112-14H.nc", "r")
wind = netCDF4.Dataset("fg_2024050112-14H.nc", "r")
prcp = netCDF4.Dataset("rr_2024050112-dummy.nc", "r")

#Finding dimensions of one of the files: The files are of the same size, here they are found of the air_temperature. The starting files for
#the different starting variables have to be of the same size to start the algorithm.
latitude = temp.variables['air_temperature'].shape[1]
longitude = temp.variables['air_temperature'].shape[2]

#writing files in a netCDF4 file
ffmcfile = netCDF4.Dataset('ffmcfile.nc', mode='w', format='NETCDF4_CLASSIC')

lat_dim = ffmcfile.createDimension('lat', latitude)  # latitude axis
lon_dim = ffmcfile.createDimension('lon', longitude)  # longitude axis
time_dim = ffmcfile.createDimension('time', None)  # unlimited axis (can be appended to).

#Attributes
ffmcfile.title = 'ffmc_test'

#Creating variables
lat = ffmcfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ffmcfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
time = ffmcfile.createVariable('time', np.float64, ('time',))
time.units = 'days from 1-5-2024'
time.long_name = 'time'
ffmc_index = ffmcfile.createVariable('ffmc', np.float64,
                                     ('time', 'lat', 'lon'))  # note: unlimited dimension is leftmost
ffmc_index.units = '1'
isi_index = ffmcfile.createVariable('isi', np.float64, ('time', 'lat', 'lon'))  # note: unlimited dimension is leftmost
isi_index.units = '1'

#Here we create the FFMC0 matrix, so we started with 85, and from here we calculate further. The base value is also set on 85
#It's easily changed to another value. The matrices will converge to a close value after about 3 days.


value = MATRIXCALC(ffmc0, longitude, latitude)
begin = value.MAKE_MAT_FFMC0()

ffmc_index[0, :, :] = begin.transpose()

isi_index[0, :, :] = begin

#Here we calculate a Matrix with the matrices of the temperature, humidity wind, percipitation and the matrix of the last
#day ffmc0.




def CALCULATE_MATRIX_FFMC(temp, rhum, wind, prcp, mat_ffmc0, day):
    longitude = temp.variables['air_temperature'].shape[1]
    latitude = temp.variables['air_temperature'].shape[2]
    i = 0
    j = 0
    mat_ffmc = np.ma.zeros((longitude, latitude))
    while i < longitude:
        while j < latitude:
            tempvalue = temp['air_temperature'][day, i, j]
            rhumvalue = rhum['relative_humidity'][day, i, j]
            windvalue = wind['windspeed'][day, i, j]
            if np.ma.is_masked(tempvalue) or np.ma.is_masked(rhumvalue) or np.ma.is_masked(windvalue):
                mat_ffmc[i][j] = np.nan #If there is no value for a value in a matrix, we call it none(this might can be done faster with a Neural Network???)
            else:
                value = FWICLASS(tempvalue, rhumvalue, windvalue, 0)
                mat_ffmc[i][j] = value.FFMCcalc(mat_ffmc0[i][j])
            j += 1
        else:
            i += 1
            j = 0
    return mat_ffmc


"""
Here we calculate the FFMC for multiple days at once, in this case 12, cause we have the data for 12 days. It takes some
time to calculate a new matrix(like 45 seconds per matrix) 
"""

i=0
while i<11:
    FFMC=ffmc_index[i, :, :]
    temperature=temp['air_temperature'][i, :, :].transpose()
    rhumidity=rhum['relative_humidity'][i, :, :].transpose()
    windspeed=wind['windspeed'][0, :, :].transpose()
    new_slice = value.MAKE_NEXT_MATRIX_FFMC(FFMC=FFMC.transpose(), temp=temperature, rhum=rhumidity, wind=windspeed)
    i+=1
    ffmc_index[i, :, :] = new_slice.transpose()

i=0
while i<11:
    FFMC=ffmc_index[i, :, :]
    temperature=temp['air_temperature'][i, :, :].transpose()
    rhumidity=rhum['relative_humidity'][i, :, :].transpose()
    windspeed=wind['windspeed'][0, :, :].transpose()
    new_slice = value.MAKE_NEXT_MATRIX_ISI(FFMC=FFMC, wind=windspeed)
    i+=1
    ffmc_index[i, :, :] = new_slice.transpose()








def CALCULATE_MATRIX_ISI(temp, rhum, wind, prcp, mat_ffmc0, day):
    longitude = temp.variables['air_temperature'].shape[1]
    latitude = temp.variables['air_temperature'].shape[2]
    i = 0
    j = 0
    mat_ffmc = ffmc_index[day, :, :]
    mat_isi = np.ma.zeros((longitude, latitude))
    while i < longitude:
        while j < latitude:
            tempvalue = temp['air_temperature'][day, i, j]
            rhumvalue = rhum['relative_humidity'][day, i, j]
            windvalue = wind['windspeed'][day, i, j]
            if np.ma.is_masked(tempvalue) or np.ma.is_masked(rhumvalue) or np.ma.is_masked(windvalue):
                mat_isi[i][j] = np.nan
            else:
                value = FWICLASS(tempvalue, rhumvalue, windvalue, 0)
                mat_isi[i][j] = value.ISIcalc(mat_ffmc[i][j])
            j += 1
        else:
            i += 1
            j = 0
    return mat_isi


"""
Here we calculate the ISI for multiple days at once, in this case 12, cause we have the data for 12 days. It takes some
time to calculate a new matrix(like 45 seconds per matrix) 

"""
#j = 0



#layers=11
#while j < layers: #till the 11th day, change for less days, or if you just want to caclulate the next one.
#    for i in tqdm(range(layers)):
#        new_slice = CALCULATE_MATRIX_ISI(temp, rhum, wind, prcp, ffmc_index[j, :, :], j)#here we caculate a ISI matrix for a new day
#        isi_index[j + 1, :, :] = new_slice #Here the new slice is written
#        j += 1 #here an we go to


def CALCULATE_MMATRIX_FFMC(temp, rhum, wind, prcp, ffmc0=85):
    days = temp.variables['air_temperature'].shape[0]
    lon = temp.variables['air_temperature'].shape[1]
    lat = temp.variables['air_temperature'].shape[2]
    mat_ffmc0 = np.full((lon, lat), ffmc0)
    i = 0
    j = 0
    day = 0
    mat_ffmc = np.ma.zeros((days, lon, lat))
    while day < days:
        while i < lon:
            while j < lat:
                tempvalue = temp['air_temperature'][day, i, j]
                rhumvalue = rhum['relative_humidity'][day, i, j]
                windvalue = wind['windspeed'][day, i, j]
                if np.ma.is_masked(tempvalue) or np.ma.is_masked(rhumvalue) or np.ma.is_masked(windvalue):
                    mat_ffmc[day][i][j] = np.nan
                else:
                    value = FWICLASS(tempvalue, rhumvalue, windvalue, 0)
                    mat_ffmc[day][i][j] = value.FFMCcalc(mat_ffmc0[i][j])
                j += 1
            else:
                i += 1
                j = 0
        else:
            mat_ffmc0 = mat_ffmc[day][:][:]
            print(str(day + 1) + "/" + str(days) + " done")
            day += 1
            i = 0
            j = 0

    return mat_ffmc


#matrix_FFMC = CALCULATE_MATRIX_FFMC(temp, rhum, wind, prcp, ffmc0_mat, 0)


#print(temp)
#print(rhum)
#print(wind)
#print(prcp)


#print(temp['air_temperature'].shape)
#plt.imshow(temp['air_temperature'][0, ::-1, :])
#plt.show()

#print(rhum['relative_humidity'].shape)
#plt.imshow(rhum['relative_humidity'][0, ::-1, :])
#plt.show()

#print(wind['windspeed'].shape)
#plt.imshow(wind['windspeed'][0, ::-1, :])
#plt.show()

#plt.imshow(matrix_FFMC[8,::-1,:])

#making the bar on the right side
#plt.colorbar()

#Displaying the title


#plt.show()


#plt.show()

#obj1 = FWICLASS(28, 17, 27, 0)
#Print the time.
print(datetime.now() - startTime)
