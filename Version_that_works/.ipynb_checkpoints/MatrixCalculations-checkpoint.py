from FireWeatherIndex import FWICLASS
import numpy as np
class MATRIXCALC:
    def __init__(self, ffmc0, long, lat):
        self.ffmc0 = ffmc0
        self.long = long
        self.lat = lat
        empty = np.ma.zeros((long, lat))
    def MAKE_MAT_FFMC0(self):
        mat_ffmc0 = np.full((self.long, self.lat), self.ffmc0)
        return mat_ffmc0

    def MAKE_MAT_EMPTY(self):
        empty = np.ma.zeros((self.long, self.lat))
        return empty

    def MAKE_NEXT_MATRIX_FFMC(self, FFMC, temp=0, rhum=0, wind=0, prcp=0):
        print()
        if temp.__class__.__name__ == 'int':
            temp = np.ma.zeros((self.long, self.lat))
            print("The temperatures are set to 0 everywhere, only use this as a test.")
        if rhum.__class__.__name__ == 'int':
            rhum = np.ma.zeros((self.long, self.lat))
            print("The relative humidity are set to 0 everywhere, only use this as a test.")
        if wind.__class__.__name__ == 'int':
            wind = np.ma.zeros((self.long, self.lat))
            print("The wind speeds are set to 0 everywhere, only use this as a test.")
        if prcp.__class__.__name__ == 'int':
            prcp = np.ma.zeros((self.long, self.lat))
            print("The precipitation are set to 0 everywhere, only use this as a test.")
        mat_ffmc = np.ma.zeros((self.long, self.lat))
        i = 0
        j = 0
        while i < self.long:
            while j < self.lat:
                if np.ma.is_masked(temp[i][j]) or np.ma.is_masked(rhum[i][j]) or np.ma.is_masked(wind[i][j]) or np.ma.is_masked(prcp[i][j]):
                    mat_ffmc[i][j] = np.nan  # If there is no value for a value in a matrix, we call it none(this
                    # might can be done faster with a Neural Network???)
                else:
                    value = FWICLASS(temp[i][j], rhum[i][j], wind[i][j], prcp[i][j])
                    mat_ffmc[i][j] = value.FFMCcalc(FFMC[i][j])
                j += 1
            else:
                i += 1
                j = 0
        return mat_ffmc

    def MAKE_NEXT_MATRIX_ISI(self, FFMC, temp=0, rhum=0, wind=0, prcp=0):
        if temp.__class__.__name__ == 'int':
            temp = np.ma.zeros((self.long, self.lat))
            print("The temperatures are set to 0 everywhere, only use this as a test.")
        if rhum.__class__.__name__ == 'int':
            rhum = np.ma.zeros((self.long, self.lat))
            print("The relative humidity are set to 0 everywhere, only use this as a test.")
        if wind.__class__.__name__ == 'int':
            wind = np.ma.zeros((self.long, self.lat))
            print("The wind speeds are set to 0 everywhere, only use this as a test.")
        if prcp.__class__.__name__ == 'int':
            prcp = np.ma.zeros((self.long, self.lat))
            print("The precipitation are set to 0 everywhere, only use this as a test.")
        mat_isi = np.ma.zeros((self.long, self.lat))
        i = 0
        j = 0
        while i < self.long:
            while j < self.lat:
                if np.ma.is_masked(temp[i][j]) or np.ma.is_masked(rhum[i][j]) or np.ma.is_masked(wind[i][j]) or np.ma.is_masked(prcp[i][j]):
                    mat_isi[i][j] = np.nan  # If there is no value for a value in a matrix, we call it none(this
                    # might can be done faster with a Neural Network???)
                else:
                    value = FWICLASS(temp[i][j], rhum[i][j], wind[i][j], prcp[i][j])
                    mat_isi[i][j] = value.ISIcalc(FFMC[i][j])
                j += 1
            else:
                i += 1
                j = 0
        return mat_isi

