import netCDF4
import numpy as np
import pyproj
import tools_utils
import os

crs = "4326"
to_srs = pyproj.Proj(init='epsg:{}'.format(crs))


class NetCDF:

    def __init__(self):
        self.NAP_water_level = 0  # default value of NAP
        self.lat = []  # latitude of dataset points
        self.lon = []  # longitude of dataset points
        self.data = []  # dataset
        return

    def read_cdffile(self, bro):
        """
        Read water level from the NHI data portal

        :param cdf_file: path to the netCDF file
        """

        # define the path for the shape file
        cdf_file = os.path.join(os.path.split(bro)[0], r"peilgebieden_jp_250m.nc")
        # open file
        dataset = netCDF4.Dataset(cdf_file)
        # read coordinates
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.lat = self.lat.ravel()
        self.lon = self.lon.ravel()
        # read data
        self.data = dataset.variables["Band1"][:].ravel()
        # close file
        dataset.close()
        return

    def query(self, X, Y):
        """
        Query data for the point X, Y

        :param X: coordinate X [RD coordinates]
        :param Y: coordinate Y [RD coordinates]
        """

        # convert to coordinate system of netCDF
        source_srs = pyproj.Proj('+init=epsg:28992'.format(crs))
        x_lon, y_lat = pyproj.transform(source_srs, to_srs, Y, X)

        # ignore errors
        np.seterr(all='ignore')

        # find index
        id_min = np.sqrt((self.lat.ravel() - x_lon)**2 - (self.lon.ravel() - y_lat)**2).argsort()

        # find the first numeric depth
        for i in id_min:
            if not str(self.data.ravel()[id_min][i]) == '--':
                self.NAP_water_level = self.data[id_min][i]
                break

        return
