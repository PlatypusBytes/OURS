
import geopandas as gpd
from shapely.geometry import Point  # You may need other geometry types depending on your data
from rtree import index

def create_idx(file_name, output_name):
    # Load shapefile
    gdf = gpd.read_file(file_name)

    # Apply buffer
    gdf['buffered_geometry'] = gdf.geometry.buffer(600)

    # Create R-tree index
    idx = index.Index(output_name, interleaved=True, overwrite=True)

    # Populate the index with buffered geometries
    for i, geom in gdf.iterrows():
        idx.insert(i, geom['buffered_geometry'].bounds)
    idx.close()
    return gdf


def is_point_inside_buffered_region():

    list_points = [[129125.375, 471341],
                   [128894, 470670],
                   [127877, 464928],
                   [131565.523, 471481.448], # False
                   [88712.725, 451981.087],
                   [88548, 452886], # False
                   [93744.64, 460834.3], # False
                   [82825.863, 455054.337]
                   ]

    idx = index.Index('./bro/buff_track', interleaved=True)

    aaa = []
    for point in list_points:
        point = Point(point[0], point[1])  # Create a shapely Point
        possible_matches = list(idx.intersection(point.bounds))
        if len(possible_matches) == 0:
            aaa.append(False)
        else:
            aaa.append(True)
    print(aaa)



if __name__ == "__main__":
    gdf = create_idx('./Auxiliary_Code/geospoortak/geosprtk.shp', './bro/buff_track')
    is_point_inside_buffered_region()
