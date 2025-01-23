import datetime
import functools
import math
import os
from copy import deepcopy
from enum import IntEnum

import numpy as np
from pysolar.solar import get_altitude
from pysolar.util import get_sunrise_sunset
from scipy.signal import convolve2d
#from WBGT_kong_huber import WBGT_Liljegren

#from lib.config import CPD_27, RO_AIR, G, R
#from lib.functions import convert_unit, operators
#from lib.logger import log

optnumba_jit_functions = {}

MINIMUM_THRESHOLD_SOLAR_RADIATION_DOWNWARDS = 0.5


############################
# GENERIC (HELPER) METHODS #
############################


def deg_to_rad(degrees: float):
    """
    Converts degrees to radians
    input in [deg]
    """
    return np.deg2rad(degrees)


def distance_in_km(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Calculates distance between two lat/lon combinations
    lat/lon [deg]
    """
    r_in_km = round(R / 1000)

    dLat = deg_to_rad(lat2 - lat1)
    dLon = deg_to_rad(lon2 - lon1)
    lat1 = deg_to_rad(lat1)
    lat2 = deg_to_rad(lat2)

    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.sin(dLon / 2) * np.sin(dLon / 2) * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = np.multiply(r_in_km, c)
    return d


def identical(an_array, **kwargs):
    return an_array


def sum_all(*args, **kwargs):
    result = None
    for arg in args:
        if result is None:
            result = arg
        else:
            result = np.add(result, arg)
    return result


def subtract_xy(x, y, **kwargs):
    """
    Return x minus y
    """
    return np.subtract(x, y)


def divide_xy(x, y, **kwargs):
    """
    Return x divided by y
    """
    return np.divide(x, y)


def multiply_xy(x, y, **kwargs):
    """
    Return x multiplied by y
    """
    return np.multiply(x, y)


def median(values: np.ndarray, **kwargs) -> np.ndarray:
    # values is a 3D array ((member, latitude, longitude) or (member,y,x))
    return np.median(values, axis=0)


def mean(values: np.ndarray, **kwargs) -> np.ndarray:
    # values is a 3D array ((member, latitude, longitude) or (member,y,x))
    return np.mean(values, axis=0)


def percentiles(values: np.ndarray, **kwargs) -> np.ndarray:
    # values is a 3D array ((member, latitude, longitude) or (member,y,x))
    percentiles = np.array(kwargs.get("percentiles", []))
    if len(percentiles) == 0 or np.any(percentiles > 100.0) or np.any(percentiles < 0.0):
        raise RuntimeError("No percentiles provided or outside valid range 0 - 100")
    return np.percentile(values, percentiles, axis=0, method=kwargs.get("method", "linear"))


def probability(values: np.ndarray, **kwargs) -> np.ndarray:
    """This method returns a 2D array of probabilities as fraction [0-1]"""
    # values is a 3D array ((member, latitude, longitude) or (member,y,x))
    threshold_operator = operators.get(kwargs.get("threshold_operator"))
    threshold_value = kwargs.get("threshold_value")
    if not threshold_operator or threshold_value is None:
        raise RuntimeError("Invalid/no threshold operator or no threshold value provided")
    return threshold_operator(values, threshold_value).sum(axis=0) / values.shape[0]


#############################
# METEOROLOGICAL ALGORITHMS #
#############################


def absolute_updraft_helicity(
    vorticity500hpa,
    vorticity600hpa,
    vorticity700hpa,
    vorticity850hpa,
    wwind500hpa,
    wwind600hpa,
    wwind700hpa,
    wwind850hpa,
    **kwargs,
):
    # The updraft helicity is computed from the formula described by Kain, J. S., and Coauthors, 2008 (Wea. Forecasting)
    heights = [1500.7, 2906.6, 4055.1, 5548.5]
    wwind = [wwind850hpa, wwind700hpa, wwind600hpa, wwind500hpa]
    vorticity = [vorticity850hpa, vorticity700hpa, vorticity600hpa, vorticity500hpa]

    updraft_helicity = np.zeros(wwind500hpa.shape)

    for level in range(len(heights) - 1):
        wwind_vorticity_mean = (wwind[level] * vorticity[level] + wwind[level + 1] * vorticity[level + 1]) / 2
        delta_height = heights[level + 1] - heights[level]
        updraft_helicity += wwind_vorticity_mean * delta_height
    return np.abs(updraft_helicity)


def cloud_area_fraction(low_cloud_area_fraction, medium_cloud_area_fraction, high_cloud_area_fraction, **kwargs):
    """
    Cloud area fraction (also in weather room referred to as progsat) calculated
    from low_cloud_area_fraction, medium_cloud_area_fraction, high_cloud_area_fraction.
    Source: algorithm: https://gitlab.com/KNMI/WLM/prentenkabinet/ppw_harmonie/-/blob/master/src/gmt/bewolking_GMT.sh
    """

    return 1.0 - (1.0 - low_cloud_area_fraction) * (1.0 - 0.7 * medium_cloud_area_fraction) * (
        1.0 - 0.5 * high_cloud_area_fraction
    )


################ COSINE SOLAR ZENITH ANGLE ######################


def _optnumba_jit(_func=None, *, nopython=True, nogil=True, parallel=True):
    """
    Code borrowed from Thermofeel 1.3.0 (https://github.com/ecmwf/thermofeel/releases/tag/1.3.0)
    """

    def decorator_optnumba(func):
        @functools.wraps(func)
        def jited_function(*args, **kwargs):
            global optnumba_jit_functions

            if func in optnumba_jit_functions:
                return optnumba_jit_functions[func](*args, **kwargs)

            if os.environ.get("THERMOFEEL_NO_NUMBA"):
                optnumba_jit_functions[func] = func
            else:
                try:
                    import numba

                    log.debug(
                        f"Numba trying to compile {func}, args: nopython {nopython} nogil {nogil} parallel {parallel}"
                    )
                    optnumba_jit_functions[func] = numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)(func)

                except Exception as e:
                    log.debug(
                        f"Numba compilation failed for {func}, reverting to pure python code -- Exception caught: {e}"
                    )
                    optnumba_jit_functions[func] = func

            assert func in optnumba_jit_functions and optnumba_jit_functions[func] is not None

            return optnumba_jit_functions[func](*args, **kwargs)

        return jited_function

    if _func is None:
        return decorator_optnumba
    else:
        return decorator_optnumba(_func)


@_optnumba_jit(parallel=False)  # function does not have benefit from parallel execution
def _cosine_solar_zenith_angle(h, lat, lon, y, m, d):
    """
    Code borrowed from Thermofeel 1.3.0 (https://github.com/ecmwf/thermofeel/releases/tag/1.3.0)

    calculate solar zenith angle
    :param h: hour [int]
    :param lat: (float array) latitude [degrees]
    :param lon: (float array) longitude [degrees]
    :param y: year [int]
    :param m: month [int]
    :param d: day [int]
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2015GL066868
    see also:
    http://answers.google.com/answers/threadview/id/782886.html
    returns cosine of the solar zenith angle (all values, including negatives)
    """
    to_radians = math.pi / 180

    # convert to julian days counting from the beginning of the year
    # jd_ = to_julian_date(d, m, y)  # julian date of data
    jd_ = (
        d
        - 32075
        + 1461 * (y + 4800 + (m - 14) / 12) / 4
        + 367 * (m - 2 - (m - 14) / 12 * 12) / 12
        - 3 * ((y + 4900 + (m - 14) / 12) / 100) / 4
    )

    # jd11_ = to_julian_date(1, 1, y)  # julian date 1st Jan
    jd11_ = (
        1
        - 32075
        + 1461 * (y + 4800 + (1 - 14) / 12) / 4
        + 367 * (1 - 2 - (1 - 14) / 12 * 12) / 12
        - 3 * ((y + 4900 + (1 - 14) / 12) / 100) / 4
    )

    jd = jd_ - jd11_ + 1  # days since start of year

    # declination angle + time correction for solar angle
    # d, tc = solar_declination_angle(jd, h)

    g = (360 / 365.25) * (jd + (h / 24))  # fractional year g in degrees
    while g > 360:
        g = g - 360
    grad = g * to_radians
    # declination in [degrees]
    d = (
        0.396372
        - 22.91327 * math.cos(grad)
        + 4.025430 * math.sin(grad)
        - 0.387205 * math.cos(2 * grad)
        + 0.051967 * math.sin(2 * grad)
        - 0.154527 * math.cos(3 * grad)
        + 0.084798 * math.sin(3 * grad)
    )
    # time correction in [ h.degrees ]
    tc = (
        0.004297
        + 0.107029 * math.cos(grad)
        - 1.837877 * math.sin(grad)
        - 0.837378 * math.cos(2 * grad)
        - 2.340475 * math.sin(2 * grad)
    )

    drad = d * to_radians

    latrad = lat * to_radians

    sindec_sinlat = np.sin(drad) * np.sin(latrad)
    cosdec_coslat = np.cos(drad) * np.cos(latrad)

    # solar hour angle [h.deg]
    sharad = ((h - 12) * 15 + lon + tc) * to_radians
    csza = sindec_sinlat + cosdec_coslat * np.cos(sharad)

    # we dont clip negative values here
    return csza


def cosine_solar_zenith_angle(h, lat, lon, y, m, d):
    """
    Returns cosine solar zenith angle
    """
    # we separate the function for clipping the negative values since numba doesn't support clip (yet)
    csza = _cosine_solar_zenith_angle(h, lat, lon, y, m, d)
    return np.clip(csza, 0, None)


##############################################################


def dew_point_from_satured_vapor_pressure(saturated_vapor_pressure, **kwargs):
    return 243.5 / ((17.57 / np.log(saturated_vapor_pressure / 6.107)) - 1)


def dew_point(temperature_celsius, relative_humidity_fraction, **kwargs):
    sat_pressure = relative_humidity_fraction * vapor_pressure(temperature_celsius)
    sat_pressure += (np.abs(sat_pressure - 6.107) < 0.0001) * sat_pressure * 0.0001
    dewpoint_temp_celsius = dew_point_from_satured_vapor_pressure(sat_pressure)
    dew_point_temp_celsius = np.maximum(temperature_celsius - 30.0, dewpoint_temp_celsius)
    # dew point can never be larger than air temperature
    return np.minimum(dew_point_temp_celsius, temperature_celsius)


def fraction_of_direct_sunlight(
    instantaneous_surface_solar_radiation_downwards, instantaneous_total_sky_direct_solar_radiation_at_surface, **kwargs
):
    """
    Fraction of direct sunlight
    """
    instantaneous_surface_solar_radiation_downwards = np.where(
        instantaneous_surface_solar_radiation_downwards < MINIMUM_THRESHOLD_SOLAR_RADIATION_DOWNWARDS,
        0,
        instantaneous_surface_solar_radiation_downwards,
    )

    fdir = np.divide(
        instantaneous_total_sky_direct_solar_radiation_at_surface,
        instantaneous_surface_solar_radiation_downwards,
        out=np.zeros_like(instantaneous_total_sky_direct_solar_radiation_at_surface),
        where=instantaneous_surface_solar_radiation_downwards != 0,
    )
    return fdir


def freezing_level_msl(freezing_level_abv_sfc_atm, geopotential_hagl_0m, **kwargs):
    """
    Calculates the height of the freezing level, or 0C isotherm, by adding the geopotential at surface (above MSL) to
    the freezing_level parameter that is related to height above the surface. This yields the height of the freezing level above MSL.

    The unit of the output is [ft]
    """
    return convert_unit(
        np.add(convert_unit(freezing_level_abv_sfc_atm, "ft", "m"), np.divide(geopotential_hagl_0m, G)), "m", "ft"
    )


def pcappi(rainconc5000ft, snowconc5000ft, grpconc5000ft, **kwargs):
    """
    Formula has been copied from prentenkabinet bash version, written by Sander Tijm
    Source: algorithm: https://gitlab.com/KNMI/WLM/prentenkabinet/ppw_harmonie/-/blob/master/src/gmt/radar_GMT_1500.sh

    We convert from mass/m^3 to a "flat" precipitation rate.
    Snow falls slower than rain, graupel has a lower density than rain.
    That is why we assign different "fall speeds" to different precipitation types.
    """

    rain_fall_speed = 5.0
    snow_fall_speed = 1.0
    grp_fall_speed = 4.0

    rain = np.where(rainconc5000ft > 0.0, rainconc5000ft * 3600 * rain_fall_speed, 0)
    snow = np.where(snowconc5000ft > 0.0, snowconc5000ft * 3600 * snow_fall_speed, 0)
    grp = np.where(grpconc5000ft > 0.0, grpconc5000ft * 3600 * grp_fall_speed, 0)

    precipitation = rain + snow + grp
    return np.where(precipitation >= 0.1, precipitation, 0)


################### PRECIPITATION TYPE #######################


class ArcusPrecipitationType(IntEnum):
    DRY = 0
    DRIZZLE = 1
    RAIN = 2
    GRAUPEL = 3
    ICE_PELLETS = 4
    HAIL = 5
    LARGE_SUMMER_HAIL = 6
    WET_SNOW = 7
    SNOW = 8
    FREEZING_DRIZZLE = 9
    FREEZING_RAIN = 10


uwcw_predominant_precipitation_type_mapping = {
    np.nan: ArcusPrecipitationType.DRY.value,
    0: ArcusPrecipitationType.DRIZZLE.value,
    1: ArcusPrecipitationType.RAIN.value,
    2: ArcusPrecipitationType.WET_SNOW.value,
    3: ArcusPrecipitationType.SNOW.value,
    4: ArcusPrecipitationType.FREEZING_DRIZZLE.value,
    5: ArcusPrecipitationType.FREEZING_RAIN.value,
    6: ArcusPrecipitationType.GRAUPEL.value,
    7: ArcusPrecipitationType.HAIL.value,
}

ecmwf_predominant_precipitation_type_mapping = {
    0: ArcusPrecipitationType.DRY.value,
    1: ArcusPrecipitationType.RAIN.value,
    3: ArcusPrecipitationType.FREEZING_RAIN.value,
    5: ArcusPrecipitationType.SNOW.value,
    6: ArcusPrecipitationType.WET_SNOW.value,
    7: ArcusPrecipitationType.WET_SNOW.value,
    8: ArcusPrecipitationType.ICE_PELLETS.value,
    12: ArcusPrecipitationType.FREEZING_DRIZZLE.value,
}


def arcus_ptype_value_from_uwcw(predominant_precipitation_type):
    return uwcw_predominant_precipitation_type_mapping.get(predominant_precipitation_type, np.nan)


def arcus_ptype_value_from_ecmwf(predominant_precipitation_type):
    return ecmwf_predominant_precipitation_type_mapping.get(predominant_precipitation_type, np.nan)


def precipitation_type(precipitation_type, total_precipitation_rate, **kwargs):
    center = kwargs.get("center", None)
    arcus_predominant_precipitation_type = precipitation_type
    if center:
        if center.lower() == "uwcw":
            vec_arcus_ptype = np.vectorize(arcus_ptype_value_from_uwcw, otypes=[np.float64])
        elif center.lower() == "ecmwf":
            vec_arcus_ptype = np.vectorize(arcus_ptype_value_from_ecmwf, otypes=[np.float64])

        arcus_predominant_precipitation_type = vec_arcus_ptype(precipitation_type)

    vec_ptype = np.vectorize(precipitation_type_value, otypes=[np.float64])
    return vec_ptype(arcus_predominant_precipitation_type, total_precipitation_rate, **kwargs)


def precipitation_type_value(precipitation_type_value, total_precipitation_rate_value, **kwargs):
    minimum_total_precipitation_rate = kwargs.get("minimum_threshold_total_precipitation_rate", 0.0)
    if total_precipitation_rate_value < minimum_total_precipitation_rate:
        precipitation_type_value = np.nan
    return precipitation_type_value


def relative_humidity(temperature_celsius, dew_point_temperature_celsius, **kwargs):
    standard_vapor_pressure = vapor_pressure(temperature_celsius)
    actual_vapor_pressure = vapor_pressure(dew_point_temperature_celsius)
    return np.clip(actual_vapor_pressure / standard_vapor_pressure, 0.0, 1.1)


################### SOLID FRACTION ###########################


def _minimum_threshold(a, threshold=None, value=None):
    if threshold is not None:
        if value is None:
            value = threshold

        a = np.where(a >= threshold, a, value)
    return a


def solid_fraction(rain, graupel, snow, **kwargs):
    if isinstance(rain, np.ndarray):
        snow = _minimum_threshold(snow, threshold=0.1, value=0.0)
        graupel = _minimum_threshold(graupel, threshold=0.1, value=0.0)
        rain = _minimum_threshold(rain, threshold=0.1, value=0.0)

        snowgraupel = np.add(snow, graupel)
        snowgraupelrain = np.add(snowgraupel, rain)
        np.seterr(invalid="ignore")
        fraction = np.clip(np.divide(snowgraupel, snowgraupelrain), 0.0, 1.0)
        fraction = np.where(np.isnan(fraction), 0.0, fraction)
    else:
        fraction = (snow + graupel) / (snow + graupel + rain)

    total = snow + graupel + rain
    return np.where(total == 0.0, np.nan, fraction)


def solid_fraction_ECMWF(snow, total, **kwargs):
    snow = _minimum_threshold(snow, threshold=0.1, value=0.0)
    total = _minimum_threshold(total, threshold=0.1, value=0.0)

    # np.seterr(invalid="ignore")
    fraction = np.clip(
        np.divide(
            snow,
            total,
            out=np.zeros_like(snow),
            where=total != 0,
        ),
        0.0,
        1.0,
    )
    fraction = np.where(np.isnan(fraction), 0.0, fraction)

    return np.where(total == 0.0, np.nan, fraction)


############ SPATIAL PROBABILITY ####################################


def spatial_probability_regular_lat_lon(parameter_values, **kwargs):
    if parameter_values.ndim == 2:
        return spatial_probability_regular_lat_lon_2D(parameter_values, **kwargs)
    result = np.zeros_like(parameter_values)
    for member in range(parameter_values.shape[0]):
        result[member] = spatial_probability_regular_lat_lon_2D(parameter_values[member], **kwargs)
    return result


def spatial_probability_regular_lat_lon_2D(parameter_values, **kwargs):
    """
    Calculates the spatial probability as a fraction with an arbitrary nearest neigbour approach:
    All gridpoints above a threshold (kwargs) within a radius (in km) divided by the sum of the potential gridpoints in range

    This function expects these variables to be set in kwargs (through derivation_scheme.yaml):
    'threshold' for example [mm/h] # HH20240702: this threshold also applies to time resolutions other than 1 hour (e.g. 3 or 6 hours for ECMWF)
    'radius_km' [km]

    Assumptions in this aproach:
    - This method currently only works for data with a regular lat/lon grid
    - Every grid point that is within radius from the center point is being used, the rest is left out. This will yield a circle that is not perfect.
    - Remember gridpoints themselves are also squares that give a value for a larger area. The net result is that the radius in the calculated grid-box circles
      varies between the set [radius] and [radius-(gridbox size in the West-East-direction)]
    - To minimize calculations the grid is divided in blocks where the first_guess_kernel_size_x is the same. It is assumed that within these blocks the constructed circle and
      contained gridpoints stay the same, in reality this is slightly changing within the block when moving from North to South, but the estimated differences are small.
    """

    output = np.zeros(parameter_values.shape)

    """
    Determine min, max, step for lat,lon
    Note: width/height are reversed by convention in numpy shape
    """
    lat_min = np.min(kwargs["latitude"])
    lat_max = np.max(kwargs["latitude"])
    step_lat = round((lat_max - lat_min) / kwargs["latitude"].shape[1], 3)
    lon_min = np.min(kwargs["longitude"])
    lon_max = np.max(kwargs["longitude"])
    step_lon = round((lon_max - lon_min) / kwargs["longitude"].shape[0], 3)

    """
    Create np array with data above threshold
    """
    mask_above_threshold = np.where(parameter_values < kwargs["threshold"], 0, 1)

    """
    # Calculate the first guess kernel size in the N-S direction. Can be calculated just once because distances in the N-S direction in a regular grid are always the same.
    """
    lon_zero_zero = kwargs["longitude"][0][0]
    lat1 = kwargs["latitude"][0][0]
    lat2 = lat1 + step_lat
    lon1 = lon2 = lon_zero_zero
    first_guess_kernel_size_y = np.add(
        1,
        np.multiply(2, np.ceil(np.divide(kwargs["radius_km"], distance_in_km(lat1, lon1, lat2, lon2)))),
    ).astype(int)

    relative_y = np.divide(np.subtract(first_guess_kernel_size_y, 1), 2).astype(int)
    """
    Calculate the array of first guess kernel sizes in the W-E direction. Has to be calculated for each row because
    the distance in the W-E direction chance in a regular grid the further away from the equator you go.
    """
    first_guess_kernel_sizes_X = np.add(
        1,
        np.multiply(
            2,
            np.ceil(
                np.divide(
                    kwargs["radius_km"],
                    distance_in_km(
                        kwargs["latitude"][:, 1],
                        lon_zero_zero,
                        kwargs["latitude"][:, 1],
                        np.add(step_lon, lon_zero_zero),
                    ),
                ),
            ),
        ),
    ).astype(int)

    changes_in_first_guess_kernel_size_x = np.unique(first_guess_kernel_sizes_X, return_index=True, return_counts=True)

    """
    To minimize calculations the grid is divided in blocks where the first_guess_kernel_size_x is the same
    # It is assumed that in these blocks the constructed circle stays the same, in reality this is slightly changing
    """
    for index in range(changes_in_first_guess_kernel_size_x[1].shape[0]):
        index_start = changes_in_first_guess_kernel_size_x[1][index]
        index_end = index_start + changes_in_first_guess_kernel_size_x[2][index]

        """
        Calculate lat at centerpoint of block; this is used to average/minimize the slight error per block
        """
        lat_centerpoint = kwargs["latitude"][int(np.floor(np.divide(np.add(index_start, index_end), 2)))][0]

        first_guess_kernel_size_x = changes_in_first_guess_kernel_size_x[0][index]
        relative_x = np.divide(np.subtract(first_guess_kernel_size_x, 1), 2).astype(int)
        """
        Determine First Guess BOX for further elimination
        """
        kernel = np.ones(
            (first_guess_kernel_size_y, first_guess_kernel_size_x)
        )  # Note: width/height are reversed by convention in numpy shape

        """
        Elimination of locations within first guess box that are outside of the given radius
        room for minor improvement:
        * split up in quadrants:
         - left top quadrant is same as the mirrored right top quadrant
         - left bottom quadrant is same as the mirrored right bottom quadrant
        * square in the centre of the circle with radius/hypotenuse could be left out for checking
        """
        for x in range(kernel.shape[1]):
            temp_lat = lat_centerpoint - (step_lat * (x - relative_x))
            for y in range(kernel.shape[0]):
                temp_lon = (step_lon * (y - relative_y)) + lon1
                if distance_in_km(lat_centerpoint, lon1, temp_lat, temp_lon) > kwargs["radius_km"]:
                    kernel[y, x] = 0

        """
        Count how much points are within the constructed circle
        """
        n_points_within_radius = (kernel > 0).sum()

        """ Calculate the amount of gridpoints within circle
        Because the convolve function is moving with a moving window that goes outside the previously defined blocks with constant first_guess_kernel_size_x
        the convolve function needs an array beyond those bounds. Therefore the bounds/offset for this function need to be set.
        """
        start_convolve = max(0, index_start - relative_y)  # Slicing doesn't work with subzero values
        offset = relative_y
        if index_start - relative_y < 0:
            offset = index_start
        end_convolve = index_end + relative_y
        n_points_above_threshold = convolve2d(
            mask_above_threshold[start_convolve:end_convolve, :],
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        result = np.where(
            n_points_above_threshold > 0,
            np.divide(n_points_above_threshold, n_points_within_radius),
            n_points_above_threshold,
        )
        """
        Now put the calculated data back into the output array at the right location.
        Note: be aware that the offset is correct when index_start-relative_y are subzero!
        """
        output[index_start:index_end, :] = result[offset : offset + (index_end - index_start), :]

    return output


####################### SUNSHINE DURATION  ##########################################


def _datetime64_to_datetime(dtg64):
    return dtg64.astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc)


def _avw_max_radiation_np(my_month, my_day, my_hour, my_lat, my_lon):
    """
    Function borrowed from the good old AVW-tool (Automatische Vizualizatie Waarnemingen) used for decades at KNMI.
    The code below is a translation from Visual Basic code obtained from Nico Maat (sept 2024).

    Calculates the maximum possible radiation for given locations and times
    based on average atmospheric composition, using numpy arrays.

    Parameters:
    my_month (np.ndarray): Array of months (1-12).
    my_day (np.ndarray): Array of days of the month.
    my_hour (np.ndarray): Array of hours of the day.
    my_lat (np.ndarray): Array of latitudes.
    my_lon (np.ndarray): Array of longitudes.

    Returns:
    np.ndarray: Array of maximum radiation values.
    """
    # Constants and initializations
    PI = np.pi
    py = PI / 180  # Conversion factor to radians

    # Calculations using numpy arrays
    d = np.floor(275 * my_month / 9) - 2 * np.floor((my_month + 9) / 12) + my_day - 30
    jd = 360 * d / 365

    sl = (jd + 279.1 + 1.9 * np.sin(jd * py)) * py
    sd = 0.398 * np.sin(sl)
    sd = np.arctan(sd / np.sqrt(-sd * sd + 1))

    se = my_lon + 15 * my_hour + 2.47 * np.sin(2 * sl)
    se = se - 1.9 * np.sin(jd * py)
    se = np.cos(se * py)

    se = np.sin(my_lat * py) * np.sin(sd) - np.cos(my_lat * py) * se * np.cos(sd)

    max_rad = 1353 * (0.6 + 0.22 * se) * se

    # Ensure non-negative result
    max_rad = np.maximum(max_rad, 0)

    return max_rad


def sunshine_duration_over_given_time_interval_avw(surface_solar_radiation_downwards_hagl_0m, **kwargs):
    """
    Calculates the hrs of sunshine since the last timestep.
    The input, surface_solar_radiation_downwards_hagl_0m should be the AVERAGE solar radiation since the past timestep in Wm-2
    It is based on the fraction of 'surface_solar_radiation_downwards_hagl_0m'/'short_wave_clear_sky_radiation_hagl_0m, the latter contains the potential SW radiation under clear sky conditions calculated
    by a function borrowed from 'AVW-tools' (Automatische Vizualizatie Waarnemingen).
    Notes:
    - The above gives a fraction that has to be corrected if since the last timestep the sun has risen or set. To do this a function from the PySolar library has been used where the angle of the sun
    under or above the horizon has been determined in order to calculate the fraction of the hour the sun was above the horizon. This fraction is multiplied with SSRD/clearsky_radiation.
    - A linear approach has been used to determine sunrise/sunset, while in reality there are small differences.
    The unit of the output is [h]
    """

    MINIMUM_SSRD = 0.2  # W.m2
    MINIMUM_POTENTIAL_CLEAR_SKY_RADIATION_PAST_INTERVAL = 0.2  # W.m2

    # to prevent SSRD from resulting in strange outcomes due to small divisions set it to zero under the MINIMUM_SSRD threshold
    surface_solar_radiation_downwards_hagl_0m = np.where(
        surface_solar_radiation_downwards_hagl_0m < MINIMUM_SSRD, 0, surface_solar_radiation_downwards_hagl_0m
    )

    lats = np.array(kwargs.get("latitude"))
    lons = np.array(kwargs.get("longitude"))

    valid_dtg = kwargs.get("valid_datetime")[0][0].astype("datetime64[s]")

    potential_clear_sky_radiation_last_interval = np.zeros_like(surface_solar_radiation_downwards_hagl_0m)
    interval = kwargs.get("time_interval_in_hours", 1)

    for hour in range(interval):
        dtg_now = _datetime64_to_datetime(valid_dtg - np.timedelta64(hour, "h"))
        dtg_moh = _datetime64_to_datetime(valid_dtg - np.timedelta64(hour + 1, "h"))

        my_month_now = datetime.datetime.fromisoformat(str(dtg_now)).month
        my_day_now = datetime.datetime.fromisoformat(str(dtg_now)).day
        my_hour_now = datetime.datetime.fromisoformat(str(dtg_now)).hour

        my_month_moh = datetime.datetime.fromisoformat(str(dtg_moh)).month
        my_day_moh = datetime.datetime.fromisoformat(str(dtg_moh)).day
        my_hour_moh = datetime.datetime.fromisoformat(str(dtg_moh)).hour

        sun_altitude_in_degrees_now = get_altitude(lats, lons, dtg_now)
        sun_altitude_in_degrees_moh = get_altitude(lats, lons, dtg_moh)

        cs_radiation_now = _avw_max_radiation_np(my_month_now, my_day_now, my_hour_now, lats, lons)
        cs_radiation_moh = _avw_max_radiation_np(my_month_moh, my_day_moh, my_hour_moh, lats, lons)
        cs_radiation_avg = np.divide(np.add(cs_radiation_now, cs_radiation_moh), 2)

        # Set fraction of sun above horizon for day and night
        fraction_sun_above_horizon = np.where(
            np.logical_and(sun_altitude_in_degrees_now > 0, sun_altitude_in_degrees_moh > 0), 1, 0
        )

        # SUNRISE - Sun above horizon now, but not last hour (sunrise)
        # In the equation below " altitude_deg_now / (altitude_deg_now - altitude_deg_minus_interval)" stands for the fraction_of_hour_sun_above_horizon, this is
        # multiplied with the clear_sky_radiation to correct for partially contribution in this hour.
        fraction_sun_above_horizon = np.where(
            np.logical_and(sun_altitude_in_degrees_now > 0, sun_altitude_in_degrees_moh <= 0),
            sun_altitude_in_degrees_now / (sun_altitude_in_degrees_now - sun_altitude_in_degrees_moh),
            fraction_sun_above_horizon,
        )

        # SUNSET - Sun below horizon now, but not last hour (sunset)
        # In the equation below "altitude_deg_minus_interval / (altitude_deg_minus_interval - altitude_deg_now)" stands for the fraction_of_hour_sun_above_horizon, this is
        # multiplied with the clear_sky_radiation to correct for partially contribution in this hour.
        fraction_sun_above_horizon = np.where(
            np.logical_and(sun_altitude_in_degrees_now <= 0, sun_altitude_in_degrees_moh > 0),
            sun_altitude_in_degrees_moh / (sun_altitude_in_degrees_moh - sun_altitude_in_degrees_now),
            fraction_sun_above_horizon,
        )

        # For ensemble arrays we need to expand the dimension of fraction_of_hour array to 3
        if surface_solar_radiation_downwards_hagl_0m.ndim == 3 and fraction_sun_above_horizon.ndim == 2:
            fraction_sun_above_horizon = np.expand_dims(fraction_sun_above_horizon, axis=0)

        # correct for partly sunshine in hour
        potential_clear_sky_radiation_last_hour = np.multiply(cs_radiation_avg, fraction_sun_above_horizon)
        potential_clear_sky_radiation_last_interval = np.add(
            potential_clear_sky_radiation_last_interval, potential_clear_sky_radiation_last_hour
        )

    # The potential_clear_sky_radiation_last_interval needs to be divided by the interval to get the average
    potential_clear_sky_radiation_last_interval = np.divide(potential_clear_sky_radiation_last_interval, interval)

    # correct for the threshold that is set for ssrd (this prevents straight jumps in the outcome)
    surface_solar_radiation_downwards_hagl_0m = np.where(
        surface_solar_radiation_downwards_hagl_0m > 0,
        surface_solar_radiation_downwards_hagl_0m - MINIMUM_SSRD,
        surface_solar_radiation_downwards_hagl_0m,
    )

    """
    To prevent division by small numbers, sunshine_fraction_since_last_interval is only calculated where potential_clear_sky_radiation_last_interval is
    above the MINIMUM_POTENTIAL_CLEAR_SKY_RADIATION_PAST_INTERVAL threshold*interval
    """
    sunshine_fraction_since_last_interval = np.divide(
        surface_solar_radiation_downwards_hagl_0m,
        potential_clear_sky_radiation_last_interval,
        out=np.zeros_like(surface_solar_radiation_downwards_hagl_0m),
        where=potential_clear_sky_radiation_last_interval
        >= interval * MINIMUM_POTENTIAL_CLEAR_SKY_RADIATION_PAST_INTERVAL,
    )

    # The sunshine_fraction_since_last_interval needs to be multiplied with the interval to get hours of sun
    sunshine_hrs_since_last_interval = np.multiply(sunshine_fraction_since_last_interval, interval)

    # clip for values more than an interval
    sunshine_hrs_since_last_interval = np.clip(sunshine_hrs_since_last_interval, 0, interval)

    return sunshine_hrs_since_last_interval


def _daylength_in_hours(valid_datetime, latitude, longitude):
    """
    return the daylength (sunset - sunrise) in fractional hours
    """
    sunrise, sunset = get_sunrise_sunset(
        latitude_deg=latitude,
        longitude_deg=longitude,
        when=valid_datetime,
    )
    return (sunset - sunrise).total_seconds() / 3600


def sunshine_percentage(sunshine_duration_past_day, valid_datetime, latitudes):
    """
    To reduce the amount of calculations, calculate the time between sunset and sunrise per latitude instead of the old
    per-gridpoint method. Could be improved slightly if old calculations for a day are stored in memory and reused for other files.
    """
    maximum_sunshine_duration = np.zeros_like(sunshine_duration_past_day)
    for index, lat in enumerate(latitudes):
        maximum_sunshine_duration[:][index] = _daylength_in_hours(valid_datetime, lat, 0)
    return np.multiply(np.clip(np.divide(sunshine_duration_past_day, maximum_sunshine_duration), 0.0, 1.0), 100)


#######################################################################


def thermal_velocity(surface_downward_sensible_heat_flux, mixed_layer_depth, **kwargs):
    """
    Calculates the velocity of thermals in the boundary layer (or mixed layer) (stijgbeweging) used for the glider forecast
    This parameter is alsow knowns as W* (wstar) or the Deardorff Velocity
    https://glossary.ametsoc.org/wiki/Deardorff_velocity?__cf_chl_tk=DThWhb5P7luM23E3bJyG3bNKm90yHn6KQ0zt9tBZU1w-1713948716-0.0.1.1-1621

    The unit of the output is [m.s-1]
    """

    """
    Constants below are taken from the prentenkabinet-script of Sander Tijm, not totally accurate, but that is not significant
    """
    g = round(G)
    Tv = 290  # Virtual temperature [K]
    ro_air = round(RO_AIR, 1)

    """
    To prevent division by (near) zero
    """
    surface_downward_sensible_heat_flux = np.where(
        surface_downward_sensible_heat_flux < 0.001, 0.001, surface_downward_sensible_heat_flux
    )

    wstar = np.power(
        np.multiply(
            np.multiply(np.divide(g, Tv), np.divide(surface_downward_sensible_heat_flux, np.multiply(CPD_27, ro_air))),
            mixed_layer_depth,
        ),
        0.33,
    )

    """
    Only values above zero are significant
    """
    wstar = np.where(wstar < 0.001, 0.001, wstar)
    return wstar


def vapor_pressure(temperature_in_celsius, **kwargs):
    return 6.112 * np.exp(17.57 * temperature_in_celsius / (temperature_in_celsius + 243.5))


def ventilation_factor(mixed_layer_depth, wind_speed_10m, **kwargs):
    """
    Calculates the ventilation factor =  windspeed * boundary layer depth [m2.s-1]
    In order to get the right units, windspeed is converted to ms-1
    Source: https://gitlab.com/KNMI/WLM/stookalert/-/blob/master/docker/src/stook_module.prep (lines 104-117)
    """
    ventilation_factor = np.multiply(mixed_layer_depth, convert_unit(wind_speed_10m, "kts", "m.s-1"))
    return np.where(ventilation_factor > 9999.999, 9999.999, ventilation_factor)


################### WEATHER CODE ###########################


class KnmiWeatherCode(IntEnum):
    CLEAR = 37

    PARTLY_CLOUDY = 8
    PARTLY_CLOUDY_DRIZZLE = 16
    PARTLY_CLOUDY_LIGHT_RAIN = 0
    PARTLY_CLOUDY_MODERATE_RAIN = 4
    PARTLY_CLOUDY_MODERATE_RAIN_THUNDER = 22
    PARTLY_CLOUDY_HEAVY_RAIN = 29
    PARTLY_CLOUDY_HEAVY_RAIN_THUNDER = 40
    PARTLY_CLOUDY_HAIL = 9
    PARTLY_CLOUDY_HAIL_THUNDER = 18
    PARTLY_CLOUDY_WET_SNOW = 34
    PARTLY_CLOUDY_WET_SNOW_THUNDER = 24
    PARTLY_CLOUDY_LIGHT_SNOW = 2
    PARTLY_CLOUDY_MODERATE_SNOW = 6
    PARTLY_CLOUDY_HEAVY_SNOW = 31
    PARTLY_CLOUDY_SNOW_THUNDER = 26

    CLOUDY = 38
    CLOUDY_DRIZZLE = 17
    CLOUDY_LIGHT_RAIN = 1
    CLOUDY_MODERATE_RAIN = 5
    CLOUDY_MODERATE_RAIN_THUNDER = 23
    CLOUDY_HEAVY_RAIN = 30
    CLOUDY_HEAVY_RAIN_THUNDER = 21
    CLOUDY_HAIL = 10
    CLOUDY_HAIL_THUNDER = 19
    CLOUDY_WET_SNOW = 35
    CLOUDY_WET_SNOW_THUNDER = 25
    CLOUDY_LIGHT_SNOW = 3
    CLOUDY_MODERATE_SNOW = 7
    CLOUDY_HEAVY_SNOW = 36
    CLOUDY_SNOW_THUNDER = 27


################### Simple weather code (hour) ###########################


def knmi_weather_code(effective_cloud_cover_fraction, total_precipitation_rate, temperature_2m, **kwargs):
    """
    Weather code vectorization

    Args:
    effective_cloud_cover_fraction (numpy.ndarray): 2D numpy array of effective cloud cover fraction (-)
    total_precipitation_rate (numpy.ndarray): 2D numpy array of total precipitation rate (mm.hr-1)
    temperature_2m (numpy.ndarray): 2D numpy array of air temperature at 2m (Celsius)

    Returns:
    weather_code (numpy.ndarray)
    """
    vec_wwcode = np.vectorize(knmi_weather_code_value, otypes=[np.float64])
    return vec_wwcode(effective_cloud_cover_fraction, total_precipitation_rate, temperature_2m, **kwargs)


def _clear(effective_cloud_cover_fraction):
    return effective_cloud_cover_fraction <= 0.2


def _partly_cloudy(effective_cloud_cover_fraction):
    return 0.2 < effective_cloud_cover_fraction <= 0.7


def _clear_or_partly_cloudy(effective_cloud_cover_fraction):
    return _clear(effective_cloud_cover_fraction) or _partly_cloudy(effective_cloud_cover_fraction)


def _cloudy(effective_cloud_cover_fraction):
    return effective_cloud_cover_fraction > 0.7


def knmi_weather_code_value(effective_cloud_cover_fraction, total_precipitation_rate, temperature_2m, **kwargs):
    """
    Simple weather code algorithm
    Source: algorithm: https://gitlab.com/KNMI/widi/processing/weerapp-data/-/blob/master/src/hourly2weathercode.py
    Fixed a bug in the source code algorithm where weather code == 5 is never returned

    Args:
    effective_cloud_cover_fraction (float): effective cloud cover fraction (-)
    total_precipitation_rate (float): total precipitation rate (mm.hr-1)
    temperature_2m (float): air temperature at 2m (Celsius)

    Returns:
    weather_code (integer)
    """

    # Sunny
    if _clear(effective_cloud_cover_fraction):
        weather_code = KnmiWeatherCode.CLEAR

    # Half Cloudy
    if _partly_cloudy(effective_cloud_cover_fraction):
        weather_code = KnmiWeatherCode.PARTLY_CLOUDY

    # Cloudy
    if _cloudy(effective_cloud_cover_fraction):
        weather_code = KnmiWeatherCode.CLOUDY

    # Sun Cloud Droplet
    if _clear_or_partly_cloudy(effective_cloud_cover_fraction) and total_precipitation_rate >= 0.1:
        weather_code = KnmiWeatherCode.PARTLY_CLOUDY_LIGHT_RAIN

    # Sun Cloud 2 Droplets
    if _clear_or_partly_cloudy(effective_cloud_cover_fraction) and total_precipitation_rate >= 3.0:
        weather_code = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_RAIN

    # Cloud Droplet
    if _cloudy(effective_cloud_cover_fraction) and total_precipitation_rate >= 0.1 and temperature_2m >= 2.0:
        weather_code = KnmiWeatherCode.CLOUDY_LIGHT_RAIN

    # Cloud 2 Droplets
    if _cloudy(effective_cloud_cover_fraction) and total_precipitation_rate >= 3.0:
        weather_code = KnmiWeatherCode.CLOUDY_MODERATE_RAIN

    # Sun Cloud Droplet Snow
    if (
        _clear_or_partly_cloudy(effective_cloud_cover_fraction)
        and total_precipitation_rate >= 0.1
        and temperature_2m < 2.0
        and temperature_2m > 0.0
    ):
        weather_code = KnmiWeatherCode.PARTLY_CLOUDY_WET_SNOW

    # Cloud Droplet Snow
    if (
        _cloudy(effective_cloud_cover_fraction)
        and total_precipitation_rate >= 0.1
        and temperature_2m < 2.0
        and temperature_2m > 0.0
    ):
        weather_code = KnmiWeatherCode.CLOUDY_WET_SNOW

    # Sun Cloud Snow
    if (
        _clear_or_partly_cloudy(effective_cloud_cover_fraction)
        and total_precipitation_rate >= 0.1
        and temperature_2m <= 0.0
    ):
        weather_code = KnmiWeatherCode.PARTLY_CLOUDY_LIGHT_SNOW

    # Sun Cloud 2 Snowflakes
    if (
        _clear_or_partly_cloudy(effective_cloud_cover_fraction)
        and total_precipitation_rate >= 3.0
        and temperature_2m <= 0.0
    ):
        weather_code = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_SNOW

    # Cloud Snow
    if _cloudy(effective_cloud_cover_fraction) and total_precipitation_rate >= 0.1 and temperature_2m <= 0.0:
        weather_code = KnmiWeatherCode.CLOUDY_LIGHT_SNOW

    # Cloud 2 Snowflakes
    if _cloudy(effective_cloud_cover_fraction) and total_precipitation_rate >= 3.0 and temperature_2m <= 0.0:
        weather_code = KnmiWeatherCode.CLOUDY_MODERATE_SNOW

    return weather_code


################### Advanced weather code (hour) ###########################


def knmi_weather_code_from_uwcw(
    predominant_precipitation_type,
    effective_cloud_cover_fraction,
    total_precipitation_rate,
    lightning,
    cape,
    **kwargs,
):
    """
    Weather code vectorization

    Args:
    predominant_precipitation_type (numpy.ndarray): 2D numpy array of effective cloud cover fraction (-)
    effective_cloud_cover_fraction (numpy.ndarray): 2D numpy array of total precipitation rate (mm.hr-1)
    total_precipitation_rate (numpy.ndarray): @D numpy array of total_precipitation_rate (mm.hr-1)
    lightning (numpy.ndarray): 2D array of lightning strikes ()
    cape (numpy.ndarray): 2D numpy array of convective available potential energy (J.kg-1)

    Thresholds & Constants used in this derivation:
    - lightning >= 1.775 :: Source: advice from Sander Tijm
    - cape >= 200.0 :: Empirical (WIK)
    - total_precipitation_rate >= 0.1 :: Empirical, generally used to separate actual precipiation from trace (WIK)

    Returns:
    weather_code (numpy.ndarray)
    """
    thunder_storm = (lightning >= 1.775) & (cape >= 200.0) & (total_precipitation_rate >= 0.1)

    vec_wwcode = np.vectorize(knmi_weather_code_value_v2, otypes=[np.float64])
    wwcode = vec_wwcode(
        predominant_precipitation_type,
        effective_cloud_cover_fraction,
        total_precipitation_rate,
        thunder_storm,
        **kwargs,
    )

    if np.isnan(wwcode).any():
        log.warning("Weather code field contains NaN's")

    return wwcode


def _drizzle(predominant_precipitation_type):
    return (ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.DRIZZLE) or (
        ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.FREEZING_DRIZZLE
    )


def _rain(predominant_precipitation_type):
    return (ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.RAIN) or (
        ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.FREEZING_RAIN
    )


def _hail(predominant_precipitation_type):
    return (ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.GRAUPEL) or (
        ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.HAIL
    )


def _wet_snow(predominant_precipitation_type):
    return ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.WET_SNOW


def _snow(predominant_precipitation_type):
    return (
        ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.ICE_PELLETS
        or ArcusPrecipitationType(predominant_precipitation_type) == ArcusPrecipitationType.SNOW
    )


def _dry(predominant_precipitation_type, total_precipitation_past_hour):
    # we include here predominant_precipitation_type for the sake of correctness. It should have NaN when
    # total_precipitation_past_hour < 0.1mm, but here we enforce 'dry' by either <0.1mm or ptype=NaN.
    return np.isnan(predominant_precipitation_type) or total_precipitation_past_hour < 0.1


def _light_precipitation(total_precipitation_past_hour):
    return 0.1 <= total_precipitation_past_hour <= 1.0


def _moderate_precipitation(total_precipitation_past_hour):
    return 1.0 < total_precipitation_past_hour < 5.0


def _light_or_moderate_precipitation(total_precipitation_past_hour):
    return _light_precipitation(total_precipitation_past_hour) or _moderate_precipitation(total_precipitation_past_hour)


def knmi_weather_code_value_v2(
    predominant_precipitation_type,
    effective_cloud_cover_fraction,
    total_precipitation_rate,
    thunder_storm,
    **kwargs,
):
    """
    Advanced weather code algorithm
    Source: weercode_icon v6.xlsx

    Args:
    predominant_precipitation_type (float): predominant precipitation type (-)
    effective_cloud_cover_fraction (float): effective_cloud_cover_fraction (0-1)
    total_precipitation_rate (float): total_precipitation_rate (mm.hr-1)
    thunderstorm (boolean): thunderstorm, true=thunderstorm, false=no-thunderstorm (-)

    Returns:
    knmi_weather_code (integer)
    """

    knmi_weather_code = np.nan
    if _dry(predominant_precipitation_type, total_precipitation_rate):
        if _clear(effective_cloud_cover_fraction):
            knmi_weather_code = KnmiWeatherCode.CLEAR.value
        elif _partly_cloudy(effective_cloud_cover_fraction):
            knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY.value
        else:
            knmi_weather_code = KnmiWeatherCode.CLOUDY.value
    else:
        if _drizzle(predominant_precipitation_type):
            if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_DRIZZLE.value
            else:
                knmi_weather_code = KnmiWeatherCode.CLOUDY_DRIZZLE.value
        elif _rain(predominant_precipitation_type):
            if thunder_storm:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    if _light_or_moderate_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_RAIN_THUNDER.value
                    else:
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_HEAVY_RAIN_THUNDER.value
                else:
                    if _light_or_moderate_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_MODERATE_RAIN_THUNDER.value
                    else:
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_HEAVY_RAIN_THUNDER.value
            else:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    if _light_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_LIGHT_RAIN.value
                    elif _moderate_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_RAIN.value
                    else:
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_HEAVY_RAIN.value
                else:
                    if _light_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_LIGHT_RAIN.value
                    elif _moderate_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_MODERATE_RAIN.value
                    else:
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_HEAVY_RAIN.value
        elif _hail(predominant_precipitation_type):
            if thunder_storm:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_HAIL_THUNDER.value
                else:
                    knmi_weather_code = KnmiWeatherCode.CLOUDY_HAIL_THUNDER.value
            else:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_HAIL.value
                else:
                    knmi_weather_code = KnmiWeatherCode.CLOUDY_HAIL.value
        elif _wet_snow(predominant_precipitation_type):
            if thunder_storm:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_WET_SNOW_THUNDER.value
                else:
                    knmi_weather_code = KnmiWeatherCode.CLOUDY_WET_SNOW_THUNDER.value
            else:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_WET_SNOW.value
                else:
                    knmi_weather_code = KnmiWeatherCode.CLOUDY_WET_SNOW.value

        elif _snow(predominant_precipitation_type):
            if thunder_storm:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_SNOW_THUNDER.value
                else:
                    knmi_weather_code = KnmiWeatherCode.CLOUDY_SNOW_THUNDER.value
            else:
                if _clear_or_partly_cloudy(effective_cloud_cover_fraction):
                    if _light_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_LIGHT_SNOW.value
                    elif _moderate_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_SNOW.value
                    else:
                        knmi_weather_code = KnmiWeatherCode.PARTLY_CLOUDY_HEAVY_SNOW.value
                else:
                    if _light_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_LIGHT_SNOW.value
                    elif _moderate_precipitation(total_precipitation_rate):
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_MODERATE_SNOW.value
                    else:
                        knmi_weather_code = KnmiWeatherCode.CLOUDY_HEAVY_SNOW.value
    return knmi_weather_code


################### Simple weather code (day) ###########################


def knmi_weather_code_past_day(
    sunshine_duration_past_day, total_precipitation_past_day, minimum_temperature_past_day, **kwargs
):
    """
    Day weather code vectorization

    Args:
    sunshine_duration_past_day (numpy.ndarray): 2D numpy array of sunshine duration over the past_day (h)
    total_precipitation_past_day (numpy.ndarray): 2D numpy array of total precipitation over the past day (mm)
    minimum_temperature_past_day (numpy.ndarray): 2D numpy array of minimum air temperature at 2m over the past day (Celsius)

    Returns:
    weather_code_past_day (numpy.ndarray)
    """
    # pySolar does NOT work with np.datetime64, so we convert to datetime
    valid_datetime = (
        kwargs.get("valid_datetime")[0][0]
        .astype("datetime64[s]")
        .astype(datetime.datetime)
        .replace(tzinfo=datetime.timezone.utc)
    )

    latitudes = kwargs.get("latitude")[:, 1]  # only get the first row

    sunshine_percentage_past_day = sunshine_percentage(sunshine_duration_past_day, valid_datetime, latitudes)

    vec_wwcode = np.vectorize(knmi_weather_code_value_past_day, otypes=[np.float64])
    return vec_wwcode(
        sunshine_percentage_past_day,
        total_precipitation_past_day,
        minimum_temperature_past_day,
        **kwargs,
    )


def knmi_weather_code_value_past_day(
    sunshine_percentage_past_day, total_precipitation_past_day, minimum_temperature_past_day, **kwargs
):
    """
    Simple weather code algorithm
    Source: algorithm: sub_maak_Meerdaagse_Forecast_Weerapp.inc.php (attached in ticket WIK-1003)
    Fixed a bug in the source code algorithm where weather code == 5 is never returned

    Args:
    sunshine_percentage_past_day (float): sunshine duration over the past_day (%)
    total_precipitation_past_day (float): total precipitation over the past day (mm)
    minimum_temperature_past_day (float): minimum air temperature at 2m over the past day (Celsius)

    Note: For total_precipitation_past_day we have chosen '0.3 mm' as significant.

    Returns:
    weather_code_past_day (integer)
    """

    # Sunny
    if sunshine_percentage_past_day >= 70.:
        weather_code_past_day = KnmiWeatherCode.CLEAR

    # Half Cloudy
    if sunshine_percentage_past_day >= 30. and sunshine_percentage_past_day < 70.:
        weather_code_past_day = KnmiWeatherCode.PARTLY_CLOUDY

    # Cloudy
    if sunshine_percentage_past_day < 30.:
        weather_code_past_day = KnmiWeatherCode.CLOUDY

    # Sun Cloud Droplet
    if sunshine_percentage_past_day >= 30. and total_precipitation_past_day >= 0.3:
        weather_code_past_day = KnmiWeatherCode.PARTLY_CLOUDY_LIGHT_RAIN

    # Sun Cloud 2 Droplets
    if sunshine_percentage_past_day >= 30. and total_precipitation_past_day >= 3.0:
        weather_code_past_day = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_RAIN

    # Cloud Droplet
    if sunshine_percentage_past_day < 30. and total_precipitation_past_day >= 0.3 and minimum_temperature_past_day >= 2.0:
        weather_code_past_day = KnmiWeatherCode.CLOUDY_LIGHT_RAIN

    # Cloud 2 Droplets
    if (
        sunshine_percentage_past_day < 30.
        and total_precipitation_past_day >= 3.0
        and minimum_temperature_past_day >= 2.0
    ):
        weather_code_past_day = KnmiWeatherCode.CLOUDY_MODERATE_RAIN

    # Sun Cloud Droplet Snow
    if (
        sunshine_percentage_past_day >= 30.
        and total_precipitation_past_day >= 0.3
        and minimum_temperature_past_day < 2.0
        and minimum_temperature_past_day > 0.0
    ):
        weather_code_past_day = KnmiWeatherCode.PARTLY_CLOUDY_WET_SNOW

    # Cloud Droplet Snow
    if (
        sunshine_percentage_past_day < 30.
        and total_precipitation_past_day >= 0.3
        and minimum_temperature_past_day < 2.0
        and minimum_temperature_past_day > 0.0
    ):
        weather_code_past_day = KnmiWeatherCode.CLOUDY_WET_SNOW

    # Sun Cloud Snow
    if (
        sunshine_percentage_past_day >= 30.
        and total_precipitation_past_day >= 0.3
        and minimum_temperature_past_day <= 0.0
    ):
        weather_code_past_day = KnmiWeatherCode.PARTLY_CLOUDY_LIGHT_SNOW

    # Sun Cloud 2 Snowflakes
    if (
        sunshine_percentage_past_day >= 30.
        and total_precipitation_past_day >= 3.0
        and minimum_temperature_past_day <= 0.0
    ):
        weather_code_past_day = KnmiWeatherCode.PARTLY_CLOUDY_MODERATE_SNOW

    # Cloud Snow
    if sunshine_percentage_past_day < 30 and total_precipitation_past_day >= 0.3 and minimum_temperature_past_day <= 0.0:
        weather_code_past_day = KnmiWeatherCode.CLOUDY_LIGHT_SNOW

    # Cloud 2 Snowflakes
    if (
        sunshine_percentage_past_day < 30.
        and total_precipitation_past_day >= 3.0
        and minimum_temperature_past_day <= 0.0
    ):
        weather_code_past_day = KnmiWeatherCode.CLOUDY_MODERATE_SNOW

    return weather_code_past_day


############# WET BULB GLOBE TEMPERATURE #################################################


def wet_bulb_globe_temperature_liljegren(
    temperature_celsius, dew_point_celsius, air_pressure, sfc_wind_speed, ssrd, fraction_of_direct_sunlight, **kwargs
):
    """
    WBGT Liljegren method (Cython implementation) based on research by Kong and Huber:
    *Kong, Qinqin, and Matthew Huber. “Explicit Calculations of Wet Bulb Globe Temperature Compared with Approximations
    and Why It Matters for Labor Productivity.” Earth’s Future, January 31, 2022. https://doi.org/10.1029/2021EF002334.*

    Usage of the Cythonized WBGT_Liljegren function:
    WBGT_Liljegren(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind)

    Expected parameters:
    # tas:      air temperature (K)
    # hurs:     relative humidity (%)
    # sfcwind:  2 meter wind speed (m/s)
    # ps:       surface pressure (Pa)
    # rsds:     surface downward solar radiation (w/m2)
    # fdir:     the ratio of direct solar radiation
    # cosz:     cosine zenith angle
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    #
    # return    outdoor wet bulb globe temperature (K)

    """
    ## Prepare parameter data

    # calculating relative humidity (calculated as fraction, therefore *100)
    relative_humidity_celcius = 100 * relative_humidity(temperature_celsius, dew_point_celsius)

    # air pressure is delivered in hPa, but Pa needed. Therefore *100
    air_pressure = 100 * air_pressure

    # temperature is deliverd in Celcius, Kelvin is needed:
    temperature_celsius = convert_unit(temperature_celsius, "C", "K")

    # getting latitude and longitude from grid
    lats = np.array(kwargs.get("latitude")).astype("float32")
    lons = np.array(kwargs.get("longitude")).astype("float32")

    # Calculation of cosine of zenith Angle
    dtg = kwargs.get("valid_datetime")[0][0].astype("datetime64[s]")
    Y, M, D, h = [dtg.astype("datetime64[%s]" % kind) for kind in "YMDh"]
    year = Y.astype(int) + 1970
    month = M.astype(int) % 12 + 1
    day = (D - M).astype(int) + 1
    hour = (h - D).astype(int)

    cossza_thrmfl = cosine_solar_zenith_angle(hour, lats, lons, year, month, day).astype("float32")

    # substituting NaN with zero
    ssrd = np.where(np.isnan(ssrd), 0, ssrd)
    ssrd = np.where(ssrd < MINIMUM_THRESHOLD_SOLAR_RADIATION_DOWNWARDS, 0, ssrd)

    # To mitigate a division by zero error near sunset when the cosine of the zenith angle is smaller than 0.05
    # every value smaller then altVal is set to this altVal

    """
    From Kong and Huber:
    Since model radiation components are stored as accumulated-over-time quantities (over each hourly interval in the case of ERA5 reanalysis data),
    the time average of cos θ during each interval is needed. However, when the accumulation intervals encompass sunset or sunrise, the inclusion
    of zeros (when the sun is below the horizon) may make the time average of cos θ too small. Being in the denominator, this too small cos θ would
    lead to an overestimation of the projected direct solar radiation and consequently too high WBGT values. A simple approximate solution to this
    problem is taking the average cos θ during only the sunlit part of each interval (Refer to Di Napoli et al. (2020) or Hogan and Hirahara (2016)
    for the calculation procedure). In Figure S1 in Supporting Information S1, we provide an example of erroneous peaks of WBGT values around sunrise
    or sunset introduced by using cos θ averaged over the whole hourly interval and also show that the peaks can be removed by averaging cos θ only
    during the sunlit period.
    """
    # Since we use an instantaneous value, there is no averaging like in the text above.
    # altVal (alternative value) is the minimum cosza value that leads to realistic values WBGT-values
    # Extensive testing learned us that with cosza values smaller than 0.05, WBGT-values spike unrealistically
    # hence altVal is set to 0.05

    altVal = 0.05
    cossza_thrmfl = np.where(cossza_thrmfl < altVal, altVal, cossza_thrmfl)

    if temperature_celsius.ndim == 3:
        wbgt_k = np.zeros_like(temperature_celsius)
        for member in range(temperature_celsius.shape[0]):
            wbgt_k[member] = WBGT_Liljegren(
                temperature_celsius[member],
                relative_humidity_celcius[member],
                air_pressure[member],
                sfc_wind_speed[member],
                ssrd[member],
                fraction_of_direct_sunlight[member],
                cossza_thrmfl,
                False,
            )
    elif temperature_celsius.ndim == 2:
        wbgt_k = WBGT_Liljegren(
            temperature_celsius,
            relative_humidity_celcius,
            air_pressure,
            sfc_wind_speed,
            ssrd,
            fraction_of_direct_sunlight,
            cossza_thrmfl,
            False,
        )
    else:
        raise RuntimeError(
            f"Temperature array of wrong shape {temperature_celsius.shape} supplied to Liljegren algorithm"
        )

    return convert_unit(wbgt_k, "K", "C")


def wet_bulb_globe_temperature_esi(
    temperature_celsius, dew_point_celsius, instantaneous_solar_radiation_downwards, **kwargs
):
    """
    Wet bulb globe temperature.
    Source: algorithm: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021EF002334
    """
    instantaneous_solar_radiation_downwards = np.where(
        np.isnan(instantaneous_solar_radiation_downwards), 0, instantaneous_solar_radiation_downwards
    )
    instantaneous_solar_radiation_downwards = np.where(
        np.isinf(instantaneous_solar_radiation_downwards), 0, instantaneous_solar_radiation_downwards
    )
    instantaneous_solar_radiation_downwards = np.where(
        instantaneous_solar_radiation_downwards < 0, 0, instantaneous_solar_radiation_downwards
    )

    instantaneous_solar_radiation_downwards = np.where(
        instantaneous_solar_radiation_downwards < MINIMUM_THRESHOLD_SOLAR_RADIATION_DOWNWARDS,
        0,
        instantaneous_solar_radiation_downwards,
    )

    relative_humidity_celsius = relative_humidity(temperature_celsius, dew_point_celsius)
    wbgt_esi = (
        0.62 * (temperature_celsius)
        - 0.7 * relative_humidity_celsius
        + 0.002 * instantaneous_solar_radiation_downwards
        + 0.43 * (temperature_celsius) * relative_humidity_celsius
        - 0.078 * pow((0.1 + instantaneous_solar_radiation_downwards), -1.0)
    )
    return wbgt_esi


def wet_bulb_potential_temperature_850hpa(temperature, relative_humidity, **kwargs):
    return wet_bulb_potential_temperature(temperature, relative_humidity, 850, **kwargs)


def wet_bulb_potential_temperature_925hpa(temperature, relative_humidity, **kwargs):
    return wet_bulb_potential_temperature(temperature, relative_humidity, 925, **kwargs)


########################################################################################


def wet_bulb_potential_temperature(temperature_celsius, relative_humidity, pressure_level, **kwargs):
    """
    Wet bulb potential temperature (also in weather room referred to as ThetaW) calculated
    from temperature, relative humidity and the associated pressure level.
    Source: algorithm: https://gitlab.com/KNMI/WLM/prentenkabinet/ppw_harmonie/-/blob/master/src/metview_macros/THW850

    Incoming temperature is in C, relative humidity is a fraction.
    """

    relative_humidity = np.clip(deepcopy(relative_humidity), 0.001, 1.0)
    temperature_kelvin = convert_unit(temperature_celsius, "C", "K")

    theta_kelvin = temperature_kelvin * np.power(1000.0 / pressure_level, 0.286)
    theta_celsius = convert_unit(theta_kelvin, "K", "C")

    dewpoint_temp_celsius = dew_point(temperature_celsius, relative_humidity)
    sat_pressure = vapor_pressure(dewpoint_temp_celsius)

    for _ in range(15):
        difference = theta_celsius - dewpoint_temp_celsius
        difference = np.maximum(0.0001, theta_celsius - dewpoint_temp_celsius)
        difference = np.minimum(difference, np.sqrt(difference))
        theta_celsius = theta_celsius - np.sqrt(0.5 * difference)
        sat_pressure += 0.6767 * np.sqrt(0.5 * difference)
        sat_pressure += (np.abs(sat_pressure - 6.107) < 0.0001) * sat_pressure * 0.0001
        dewpoint_temp_celsius = dew_point_from_satured_vapor_pressure(sat_pressure)

    return theta_celsius


def wind_chill(temperature, wind_speed, **kwargs):
    """
    Formula has been copied from prentenkabinet bash version, written by Sander Tijm
    Source: algorithm: https://gitlab.com/KNMI/WLM/prentenkabinet/ppw_harmonie/-/blob/master/src/metview_macros/T_GEVOEL

    # TG=13.12+0.6215*(T2M-273.15)-11.37*(3.6*WIND10)**0.16+0.3965*(T2M-273.15)*(3.6*WIND10)**0.16

    # This is an emperical formula based on specific units
    """
    wind_kmh = convert_unit(wind_speed, "kts", "km.h-1")

    a = 0.6215 * temperature
    b = 11.37 * np.power(wind_kmh, 0.16)
    c = 0.3965 * temperature * np.power(wind_kmh, 0.16)

    windchill = 13.12 + a - b + c
    return windchill


def wind_direction(u, v, **kwargs):
    wind_dir = np.arctan2(-u, -v) * 180 / np.pi
    return np.where(wind_dir < 0, wind_dir + 360, wind_dir)


def wind_shear(u_level1, v_level1, u_level2, v_level2, **kwargs):
    delta_u = u_level1 - u_level2
    delta_v = v_level1 - v_level2
    return wind_speed(delta_u, delta_v)


def wind_speed(u, v, **kwargs):
    return np.sqrt(np.square(u) + np.square(v))


def wstar_ustar_ratio(turbulent_kinetic_energy, thermal_velocity, **kwargs):
    """
    Calculates W*/U* from the turbulent kinetic energy in the lowest model level and Thermal Velocity (or W* or Deardorff Velocity)
    Ustar is the average friction velocity [ms-1]
    This algorithm is copied from Sander Tijms prentenkabinet-script, he refered to unknown literature used at Wageningen University (likely Stull).

    The unit of the output is dimensionless [-]
    """
    ustar = np.divide((turbulent_kinetic_energy - 0.2 * np.power(thermal_velocity, 2)), 3.75)
    ustar = np.where(ustar < 0.001, 0.001, ustar)
    wstar_over_ustar = np.divide(thermal_velocity, ustar)
    return np.where(wstar_over_ustar >= 10, 10, wstar_over_ustar)
