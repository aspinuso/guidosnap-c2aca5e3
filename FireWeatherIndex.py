"""
This file makes fire indeces from a file of the Canadian Forest Service. This file has multiple indeces all culminating
in the Fire Weather Index(FWI) and can be found here:
https://ostr-backend-prod.azurewebsites.net/server/api/core/bitstreams/08dbea89-c262-4009-95ca-dd74bf3f66cc/content
The file has code for multiple types of programming languages made from original code in Fortran, where here the python
code is almost directly copied to make maps of this code.
The quantities in the code are listed in page 2 of this file and will be very useful
The Python code done here is written from page 14 to 16.
FFMC and ISI has maps calculated in other files, for other indeces more info is often needed. Which is also explained in
the file.

Guido Reitsma
"""
import math

"""Define Class FWI"""


class FWICLASS:
    def __init__(self, temp, rhum, wind, prcp):
        self.h = rhum  #initializes relative humidity#
        self.t = temp  #initializes temperature#
        self.w = wind  #initializes wind#
        self.p = prcp  #initializes percipitation#
 
    def FFMCcalc(self, ffmc0):
        mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0)
        if (self.p > 0.5):
            rf = self.p - 0.5
            if (mo > 150.0):
                mo = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))) \
                     + (.0015 * (mo - 150.0) ** 2) * math.sqrt(rf)
            elif mo <= 150.0:
                mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
            if (mo > 250.0):
                mo = 250.0
        ed = .942 * (self.h ** .679) + (11.0 * math.exp((self.h - 100.0) / 10.0)) + 0.18 * (21.1 - self.t) \
             * (1.0 - 1.0 / math.exp(.1150 * self.h))
        if (mo < ed):
            ew = .618 * (self.h ** .753) + (10.0 * math.exp((self.h - 100.0) / 10.0)) \
                + .18 * (21.1 - self.t) * (1.0 - 1.0 / math.exp(.115 * self.h))
            if (mo <= ew):
                kl = .424 * (1.0 - ((100.0 - self.h) / 100.0) ** 1.7) + (.0694 * math.sqrt(self.w)) \
                     * (1.0 - ((100.0 - self.h) / 100.0) ** 8)
                kw = kl * (.581 * math.exp(.0365 * self.t))
                m = ew - (ew - mo) / 10.0 ** kw
            elif (mo > ew):
                m = mo
        elif (mo == ed):
            m = mo
        elif (mo > ed):
            kl = .424 * (1.0 - (self.h / 100.0) ** 1.7) + (.0694 * math.sqrt(self.w)) * \
                 (1.0 - (self.h / 100.0) ** 8)
            kw = kl * (.581 * math.exp(.0365 * self.t))
            m = ed + (mo - ed) / 10.0 ** kw

        ffmc =  (59.5 * (250.0 -m)) / (147.2 + m)
        if (ffmc > 101.0):
            ffmc = 101.0
        if (ffmc <= 0.0):
            ffmc = 0.0
        return ffmc

    def DMCcalc(self,dmc0,mth):
        le = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0] # EFFECTIVE DAY LENGTH GIVEN THE
        # MONTH! Changes by place, maybe make a calculation given the latitude? Also changed el in le, like the file.
        t = self.t
        if (t < -1.1):
            t = -1.1
        rk = 1.894 * (t + 1.1) * (100.0 - self.h) * (le[mth - 1] * 0.0001)
        if self.p > 1.5:
            ra = self.p
            rw = 0.92 * ra - 1.27
            wmi = 20.0 + 280.0 / math.exp(0.023 * dmc0)
            if dmc0 <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc0)
            elif dmc0 > 33.0:
                if dmc0 <= 65.0:
                    b = 14.0 - 1.3 * math.log(dmc0)
                elif dmc0 > 65.0:
                    b = 6.2 * math.log(dmc0) - 17.2
            wmr = wmi + (1000 * rw) / (48.77 + b * rw)
            pr = 43.43 * (5.6348 - math.log(wmr - 20.0))
        elif self.p <= 1.5:
            pr = dmc0
        if (pr < 0.0):
            pr = 0.0
        dmc = pr + rk
        if (dmc <= 1.0):
            dmc = 1.0
        return dmc

    def ISIcalc(self,ffmc):
        mo = 147.2 * (101.0-ffmc) / (59.5+ffmc)
        ff = 19.115 * math.exp(mo * -0.1386) * (1.0 + (mo ** 5.31) / 49300000.0)
        isi = ff * math.exp(0.05039 * self.w)
        return isi

    def DCcalc(self,dc0,mth):
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]#I Don't know what values we need here, so for now I will keep it this way
        t = self.t
        if (t < -2.8):
            t = -2.8
        pe = (0.36 * (t + 2.8) + fl[mth - 1]) / 2
        if pe <= 0.0:
            pe = 0.0
        if (self.p > 2.8):
            ra = self.p
            rw = 0.83 * ra - 1.27  # *Eq. 18*#
            smi = 800.0 * math.exp(-dc0 / 400.0)  # *Eq. 19*#
            dr = dc0 - 400.0 * math.log(1.0 + ((3.937 * rw) / smi))  # *Eqs. 20 and 21*#
            if (dr > 0.0):
                dc = dr + pe
            else:
                dc = pe
        elif self.p <= 2.8:
            dc = dc0 + pe
        return dc

    def BUIcalc(self,dmc,dc):
        if dmc <= 0.4 * dc:
            bui = (0.8 * dc * dmc) / (dmc + 0.4 * dc)
        else:
            bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
        if bui < 0.0:
            bui = 0.0
        return bui

    def FWIcalc(self,isi,bui):
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
        else:
            bb = 0.1 * isi * (1000.0 / (25. + 108.64 / math.exp(0.023 * bui)))
        if (bb <= 1.0):
            fwi = bb
        else:
            fwi = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)
        return fwi
#A new possible way to calculate indices with the FFMC directly instead of ISI
    def FWIcalcffmc(self,ffmc,bui):
        mo = 147.2 * (101.0-ffmc) / (59.5+ffmc)
        ff = 19.115 * math.exp(mo * -0.1386) * (1.0 + (mo ** 5.31) / 49300000.0)
        ffmc_exp = ff * math.exp(0.5039)
        if bui <= 80.0:
            bb = 0.1 * ffmc_exp * (0.626 * bui ** 0.809 + 2.0)
        else:
            bb = 0.1 * ffmc_exp * (1000.0 / (25. + 108.64 / math.exp(0.023 * bui)))
        if (bb <= 1.0):
            ffmc_exp = bb
        else:
            fwi = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)
        return fwi
        

    




