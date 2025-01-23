from FireWeatherIndex import FWICLASS
import numpy as np
class MATRIXCALC:
    def __init__(self, long, lat, ffmc0=85, dc0=15, dmc0=6):
        self.ffmc0 = ffmc0
        self.dc0 = dc0
        self.dmc0 = dmc0
        self.long = long
        self.lat = lat
        empty = np.ma.zeros((long, lat))
        
    def MAKE_MAT_FFMC0(self):
        mat_ffmc0 = np.full((self.long, self.lat), self.ffmc0)
        return mat_ffmc0

    def MAKE_MAT_DC0(self):
        mat_ffmc0 = np.full((self.long, self.lat), self.dc0)
        return mat_ffmc0

    def MAKE_MAT_DMC0(self):
        mat_ffmc0 = np.full((self.long, self.lat), self.dmc0)
        return mat_ffmc0   
        
    def MAKE_MAT_EMPTY(self):
        empty = np.ma.zeros((self.long, self.lat))
        return empty

    def MAKE_NEXT_MATRIX_FFMC(self, FFMC, temp=0, rhum=0, wind=0, prcp=0):
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
        if rhum.__class__.__name__ == 'int':
            rhum = np.ma.zeros((self.long, self.lat))
        if wind.__class__.__name__ == 'int':
            wind = np.ma.zeros((self.long, self.lat))
            print("The wind speeds are set to 0 everywhere, only use this as a test.")
        if prcp.__class__.__name__ == 'int':
            prcp = np.ma.zeros((self.long, self.lat))
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

    def MAKE_NEXT_MATRIX_DMC(self, DMC, month, temp=0, rhum=0, wind=0, prcp=0):
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
        mat_dmc = np.ma.zeros((self.long, self.lat))
        i = 0
        j = 0
        while i < self.long:
            while j < self.lat:
                if np.ma.is_masked(temp[i][j]) or np.ma.is_masked(rhum[i][j]) or np.ma.is_masked(wind[i][j]) or np.ma.is_masked(prcp[i][j]):
                    mat_dmc[i][j] = np.nan  # If there is no value for a value in a matrix, we call it none(this
                    # might can be done faster with a Neural Network???)
                else:
                    value = FWICLASS(temp[i][j], rhum[i][j], wind[i][j], prcp[i][j])
                    mat_dmc[i][j] = value.DMCcalc(DMC[i][j], month)
                j += 1
            else:
                i += 1
                j = 0
        return mat_dmc

    def MAKE_NEXT_MATRIX_DC(self, DC, month, temp=0, rhum=0, wind=0, prcp=0):
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
        mat_dc = np.ma.zeros((self.long, self.lat))
        i = 0
        j = 0
        while i < self.long:
            while j < self.lat:
                if np.ma.is_masked(temp[i][j]) or np.ma.is_masked(rhum[i][j]) or np.ma.is_masked(wind[i][j]) or np.ma.is_masked(prcp[i][j]):
                    mat_dc[i][j] = np.nan  # If there is no value for a value in a matrix, we call it none(this
                    # might can be done faster with a Neural Network???)
                else:
                    value = FWICLASS(temp[i][j], rhum[i][j], wind[i][j], prcp[i][j])
                    mat_dc[i][j] = value.DCcalc(DC[i][j], month)
                j += 1
            else:
                i += 1
                j = 0
        return mat_dc

    def MAKE_NEXT_MATRIX_BUI(self, DMC, DC, temp=0, rhum=0, wind=0, prcp=0):
        if temp.__class__.__name__ == 'int':
            temp = np.ma.zeros((self.long, self.lat))
        if rhum.__class__.__name__ == 'int':
            rhum = np.ma.zeros((self.long, self.lat))
        if wind.__class__.__name__ == 'int':
            wind = np.ma.zeros((self.long, self.lat))
        if prcp.__class__.__name__ == 'int':
            prcp = np.ma.zeros((self.long, self.lat))
        mat_bui = np.ma.zeros((self.long, self.lat))
        i = 0
        j = 0
        while i < self.long:
            while j < self.lat:
                if np.ma.is_masked(temp[i][j]) or np.ma.is_masked(rhum[i][j]) or np.ma.is_masked(wind[i][j]) or np.ma.is_masked(prcp[i][j]):
                    mat_bui[i][j] = np.nan  # If there is no value for a value in a matrix, we call it none(this
                    # might can be done faster with a Neural Network???)
                else:
                    value = FWICLASS(temp[i][j], rhum[i][j], wind[i][j], prcp[i][j])
                    mat_bui[i][j] = value.BUIcalc(DMC[i][j],DC[i][j])
                j += 1
            else:
                i += 1
                j = 0
        return mat_bui

    def MAKE_NEXT_MATRIX_FWI(self, ISI, BUI, temp=0, rhum=0, wind=0, prcp=0):
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
        mat_fwi = np.ma.zeros((self.long, self.lat))
        i = 0
        j = 0
        while i < self.long:
            while j < self.lat:
                if np.ma.is_masked(temp[i][j]) or np.ma.is_masked(rhum[i][j]) or np.ma.is_masked(wind[i][j]) or np.ma.is_masked(prcp[i][j]):
                    mat_fwi[i][j] = np.nan  # If there is no value for a value in a matrix, we call it none(this
                    # might can be done faster with a Neural Network???)
                else:
                    value = FWICLASS(temp[i][j], rhum[i][j], wind[i][j], prcp[i][j])
                    mat_fwi[i][j] = value.FWIcalc(ISI[i][j],BUI[i][j])
                j += 1
            else:
                i += 1
                j = 0
        return mat_fwi

