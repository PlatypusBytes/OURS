from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch


class Create_Shape_File:
    def __init__(self):
        self.soil_type_1 = []
        self.soil_type_2 = []
        self.soil_type_3 = []
        self.soil_type_4 = []
        self.soil_type_5 = []
        self.soil_type_6 = []
        self.soil_type_7 = []
        self.soil_type_8 = []
        self.soil_type_9 = []

        self.poligon_1 = []
        self.poligon_2 = []
        self.poligon_3 = []
        self.poligon_4 = []
        self.poligon_5 = []
        self.poligon_6 = []
        self.poligon_7 = []
        self.poligon_8 = []
        self.poligon_9 = []

        return

    def soil_types_robertson(self):
        r"""
        Defines geometries for the soil types, following Robertson and Cabal :cite:`robertson_cabal_2014`.
        """
        import numpy as np
        from shapely.geometry import Point, Polygon

        line1 = [[0.1, 1.],
                 [0.1, 10.]]

        line2 = [[0.1, 1.],
                 [1.67874, 1.]]

        line3 = [[0.1, 9.9087],
                 [0.100786, 9.9075],
                 [0.149174, 9.7668],
                 [0.161125, 9.7011],
                 [0.20994, 9.3068],
                 [0.22365, 9.1669],
                 [0.31112, 8.1925]]

        line4 = [[0.31112, 8.1925],
                 [0.32019, 8.0924],
                 [0.34937, 7.7779],
                 [0.4222, 7.0482],
                 [0.49102, 6.4297],
                 [0.5864, 5.6589],
                 [0.61785, 5.4228],
                 [0.67012, 5.0478],
                 [0.73322, 4.6227]]

        line5 = [[0.73322, 4.6227],
                 [0.84239, 3.956],
                 [0.94826, 3.3915],
                 [1.02828, 3.0142],
                 [1.11172, 2.6607],
                 [1.12577, 2.6049],
                 [1.19338, 2.3499],
                 [1.34843, 1.8446],
                 [1.38716, 1.7336],
                 [1.49254, 1.4559],
                 [1.63011, 1.11874]]

        line6 = [[1.63011, 1.11874],
                 [1.67874, 1.]]

        line7 = [[0.1, 9.9087],
                 [0.1, 26.227]]

        line8 = [[0.1, 26.227],
                 [0.100786, 26.242],
                 [0.149174, 27.388],
                 [0.161125, 27.746],
                 [0.20994, 29.345],
                 [0.22365, 29.818],
                 [0.31112, 33.021],
                 [0.32019, 33.373],
                 [0.34937, 34.527],
                 [0.4222, 37.518],
                 [0.49102, 40.43],
                 [0.5864, 44.572],
                 [0.61785, 45.968],
                 [0.67012, 48.338],
                 [0.73322, 51.312],
                 [0.84239, 56.865],
                 [0.94826, 62.789],
                 [1.02828, 67.551],
                 [1.11172, 72.73],
                 [1.12577, 73.622],
                 [1.19338, 77.993],
                 [1.34843, 88.533],
                 [1.38716, 91.294],
                 [1.49254, 99.22],
                 [1.63011, 111.066],
                 [1.67874, 115.836],
                 [1.7354, 121.861],
                 [1.8392, 134.27],
                 [1.8632, 137.36],
                 [1.9537, 149.75],
                 [2.1434, 178.77],
                 [2.4529, 231.22]]

        line9 = [[0.31112, 8.1925],
                 [0.32019, 8.2447],
                 [0.34937, 8.5967],
                 [0.4222, 9.4364],
                 [0.49102, 10.2107],
                 [0.5864, 11.2906],
                 [0.61785, 11.6524],
                 [0.67012, 12.2604],
                 [0.73322, 13.004],
                 [0.84239, 14.311],
                 [0.94826, 15.607],
                 [1.02828, 16.606],
                 [1.11172, 17.668],
                 [1.12577, 17.849],
                 [1.19338, 18.732],
                 [1.34843, 20.828],
                 [1.38716, 21.369],
                 [1.49254, 22.88],
                 [1.63011, 24.951],
                 [1.67874, 25.713],
                 [1.7354, 26.622],
                 [1.8392, 28.355],
                 [1.8632, 28.767],
                 [1.9537, 30.375],
                 [2.1434, 34.017],
                 [2.4529, 40.905],
                 [2.4819, 41.62],
                 [2.5315, 42.875],
                 [2.6649, 46.454],
                 [2.9778, 56.127],
                 [3.0694, 59.299],
                 [3.3499, 69.802],
                 [3.8111, 99.172],
                 [3.8336, 97.603]]

        line10 = [[2.4529, 231.22],
                  [2.4819, 231.81],
                  [2.5315, 221.29],
                  [2.6649, 196.47],
                  [2.9778, 154.95],
                  [3.0694, 145.94],
                  [3.3499, 123.626],
                  [3.8111, 99.172],
                  [3.8336, 97.603]]

        line11 = [[0.73322, 4.6227],
                  [0.84239, 5.0455],
                  [0.94826, 5.4446],
                  [1.02828, 5.7439],
                  [1.11172, 6.0567],
                  [1.12577, 6.1096],
                  [1.19338, 6.3654],
                  [1.34843, 6.9646],
                  [1.38716, 7.118],
                  [1.49254, 7.5453],
                  [1.63011, 8.1286],
                  [1.67874, 8.3428],
                  [1.7354, 8.5979],
                  [1.8392, 9.083],
                  [1.8632, 9.1982],
                  [1.9537, 9.6458],
                  [2.1434, 10.6487],
                  [2.4529, 12.4642],
                  [2.4819, 12.645],
                  [2.5315, 12.96],
                  [2.6649, 13.833],
                  [2.9778, 16.045],
                  [3.0694, 16.738],
                  [3.3499, 19.004],
                  [3.8111, 23.315],
                  [3.8336, 23.548],
                  [4.2133, 27.903],
                  [4.2428, 28.279],
                  [4.412, 30.568],
                  [4.6227, 33.761],
                  [4.6535, 34.261],
                  [4.731, 35.567],
                  [5.1431, 43.631],
                  [5.955, 65.551],
                  [5.955, 65.552]]

        line12 = [[3.8336, 97.603],
                  [4.2133, 85.157],
                  [4.2428, 84.455],
                  [4.412, 81.032],
                  [4.6227, 77.865],
                  [4.6535, 77.472],
                  [4.731, 76.538],
                  [5.1431, 72.468],
                  [5.955, 66.606],
                  [5.955, 65.552]]

        line13 = [[1.63011, 1.11874],
                  [1.67874, 1.13308],
                  [1.7354, 1.14917],
                  [1.8392, 1.17843],
                  [1.8632, 1.18518],
                  [1.9537, 1.21084],
                  [2.1434, 1.2662],
                  [2.4529, 1.363],
                  [2.4819, 1.3726],
                  [2.5315, 1.3891],
                  [2.6649, 1.4347],
                  [2.9778, 1.5484],
                  [3.0694, 1.5833],
                  [3.3499, 1.6954],
                  [3.8111, 1.8963],
                  [3.8336, 1.9066],
                  [4.2133, 2.0901],
                  [4.2428, 2.105],
                  [4.412, 2.1928],
                  [4.6227, 2.3067],
                  [4.6535, 2.3237],
                  [4.731, 2.3671],
                  [5.1431, 2.6082],
                  [5.955, 3.1333],
                  [5.955, 3.1334],
                  [6.7445, 3.7116],
                  [7.6204, 4.4719],
                  [8.3793, 5.359],
                  [9.8663, 7.7562],
                  [10., 7.9877]]

        line14 = [[5.955, 65.552],
                  [6.7445, 62.224],
                  [7.6204, 59.12],
                  [8.3793, 57.827],
                  [9.8663, 56.834],
                  [10., 56.779]]

        line15 = [[10., 56.779],
                  [10., 7.9877]]

        line16 = [[1.67874, 1.],
                  [10., 1.]]

        line17 = [[10., 1.],
                  [10., 7.9877]]

        line18 = [[0.1, 26.227],
                  [0.1, 151.61]]

        line19 = [[0.1, 151.61],
                  [0.100786, 151.77],
                  [0.149174, 163],
                  [0.161125, 166.4],
                  [0.20994, 183.08],
                  [0.22365, 188.3],
                  [0.31112, 223.36],
                  [0.32019, 227.11],
                  [0.34937, 239.4],
                  [0.4222, 273],
                  [0.49102, 311.67],
                  [0.5864, 376.71],
                  [0.61785, 400.58],
                  [0.67012, 442.95],
                  [0.73322, 499.9],
                  [0.84239, 629.43],
                  [0.94826, 815.18],
                  [1.02828, 1000.]]

        line20 = [[1.02828, 1000],
                  [1.38716, 1000.]]

        line21 = [[1.38716, 1000],
                  [1.49254, 826.09],
                  [1.63011, 651.79],
                  [1.67874, 603.11],
                  [1.7354, 553.11],
                  [1.8392, 476.75],
                  [1.8632, 461.5],
                  [1.9537, 410.3],
                  [2.1434, 327.82],
                  [2.4529, 231.22]]

        line22 = [[0.1, 151.61],
                  [0.1, 1000.]]

        line23 = [[0.1, 1000.],
                  [1.02828, 1000.]]

        line24 = [[3.8336, 97.603],
                  [4.2133, 167.95],
                  [4.2428, 175.32],
                  [4.412, 229.21],
                  [4.6227, 455.19],
                  [4.6535, 571.73],
                  [4.731, 1000.]]

        line25 = [[1.38716, 1000.],
                  [4.731, 1000.]]

        line26 = [[10., 56.779],
                  [10., 1000.]]

        line27 = [[4.731, 1000.],
                  [10., 1000.]]

        # soil types
        self.soil_type_1 = np.concatenate((np.array(line1), np.array(line3), np.array(line4), np.array(line5), np.array(line6), np.array(line2)), axis=0)
        self.soil_type_2 = np.concatenate((np.array(line6), np.array(line13), np.array(line17), np.array(line16)[::-1]), axis=0)
        self.soil_type_3 = np.concatenate((np.array(line5)[::-1], np.array(line11), np.array(line14), np.array(line15), np.array(line13)[::-1]), axis=0)
        self.soil_type_4 = np.concatenate((np.array(line4)[::-1], np.array(line9), np.array(line12), np.array(line11)[::-1]), axis=0)
        self.soil_type_5 = np.concatenate((np.array(line3[::-1]), np.array(line7), np.array(line8), np.array(line10), np.array(line9)[::-1]), axis=0)
        self.soil_type_6 = np.concatenate((np.array(line18), np.array(line19), np.array(line20), np.array(line21), np.array(line8)[::-1]), axis=0)
        self.soil_type_7 = np.concatenate((np.array(line22), np.array(line23), np.array(line19)[::-1]), axis=0)
        self.soil_type_8 = np.concatenate((np.array(line21), np.array(line25), np.array(line24)[::-1], np.array(line10)[::-1]), axis=0)
        self.soil_type_9 = np.concatenate((np.array(line12)[::-1], np.array(line24), np.array(line27), np.array(line26)[::-1], np.array(line14)[::-1]), axis=0)

        # poligon
        coords = [(i[0], i[1]) for i in self.soil_type_1]
        self.poligon_1 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_2]
        self.poligon_2 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_3]
        self.poligon_3 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_4]
        self.poligon_4 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_5]
        self.poligon_5 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_6]
        self.poligon_6 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_7]
        self.poligon_7 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_8]
        self.poligon_8 = Polygon(coords)
        coords = [(i[0], i[1]) for i in self.soil_type_9]
        self.poligon_9 = Polygon(coords)

        return


    def create_shape_file(self):
        import shapefile
        self.soil_types_robertson()
        polygons_list = [Rb.poligon_1, Rb.poligon_2, Rb.poligon_3, Rb.poligon_4, Rb.poligon_5, Rb.poligon_6,
                         Rb.poligon_7, Rb.poligon_8, Rb.poligon_9]


        w = shapefile.Writer('shapefiles/Robertson')
        w.field('name', 'C')
        final_list = []
        counter = 0
        for i in polygons_list:
            final_list.append(list(zip(i.exterior.xy[0], i.exterior.xy[1])))
            w.poly([list(zip(i.exterior.xy[0], i.exterior.xy[1]))])
            w.record('polygon' + str(counter))
            counter = counter + 1
        w.close()

        return


def plot_shape_file(path):
    # Plot Shape file
    import shapefile
    sf = shapefile.Reader(path)
    print('number of shapes imported:', len(sf.shapes()))

    plt.figure()
    ax = plt.axes()
    for shape in list(sf.iterShapes()):
        x_lon, y_lat = zip(*shape.points)
        plt.plot(x_lon, y_lat)
    plt.show()
    return


# plot_shape_file('shapefiles/Robertson')
