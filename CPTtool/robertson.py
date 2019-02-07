class Robertson:
    r"""
    Robertson soil classification.

    Classification of soils according to Robertson chart.

    .. _element:
    .. figure:: ./_static/robertson.png
        :width: 350px
        :align: center
        :figclass: align-center
    """

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

    def soil_types(self, path_shapefile=r"./shapefiles/" , model_name = 'Robertson'):
        import shapefile
        from shapely.geometry import Polygon
        import os
        import sys

        sf = shapefile.Reader(path_shapefile+model_name)
        self.poligon_1 = Polygon(list(sf.iterShapes())[0].points)
        self.poligon_2 = Polygon(list(sf.iterShapes())[1].points)
        self.poligon_3 = Polygon(list(sf.iterShapes())[2].points)
        self.poligon_4 = Polygon(list(sf.iterShapes())[3].points)
        self.poligon_5 = Polygon(list(sf.iterShapes())[4].points)
        self.poligon_6 = Polygon(list(sf.iterShapes())[5].points)
        self.poligon_7 = Polygon(list(sf.iterShapes())[6].points)
        self.poligon_8 = Polygon(list(sf.iterShapes())[7].points)
        self.poligon_9 = Polygon(list(sf.iterShapes())[8].points)

        return

    def lithology(self, Qtn, Fr):
        r"""
        Identifies lithology of CPT points, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        Parameters
        ----------
        :param gamma_limit: Maximum value for gamma
        :param z_pwp: Depth pore water pressure
        :param iter_max: (optional) Maximum number of iterations
        :return: lithology array, Qtn, Fr
        """
        import numpy as np
        from shapely.geometry import Point

        litho = [""] * len(Qtn)
        coords = np.zeros((len(Qtn), 2))

        # determine into which soil type the point is
        for i in range(len(Qtn)):
            pnt = Point(Fr[i], Qtn[i])

            aux = [self.poligon_1.contains(pnt),
                   self.poligon_2.contains(pnt),
                   self.poligon_3.contains(pnt),
                   self.poligon_4.contains(pnt),
                   self.poligon_5.contains(pnt),
                   self.poligon_6.contains(pnt),
                   self.poligon_7.contains(pnt),
                   self.poligon_8.contains(pnt),
                   self.poligon_9.contains(pnt),
                   ]
            # check if point is within a boundary
            if all(not x for x in aux):
                aux = [self.poligon_1.touches(pnt),
                       self.poligon_2.touches(pnt),
                       self.poligon_3.touches(pnt),
                       self.poligon_4.touches(pnt),
                       self.poligon_5.touches(pnt),
                       self.poligon_6.touches(pnt),
                       self.poligon_7.touches(pnt),
                       self.poligon_8.touches(pnt),
                       self.poligon_9.touches(pnt),
                       ]

            idx = np.where(np.array(aux))[0][0]
            litho[i] = str(idx + 1)
            coords[i] = [Fr[i], Qtn[i]]

        return litho, np.array(coords)
