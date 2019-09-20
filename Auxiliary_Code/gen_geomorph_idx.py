from rtree import index
import fiona
from shapely.geometry import shape
import pprint
from tqdm import tqdm


def geomorph_to_idx(shp_fn):
    idx = index.Index("geomorph", interleaved=True, overwrite=True)
    with fiona.open(shp_fn, 'r') as shp:
        pprint.pprint(shp.schema)
        for (i, polygon) in tqdm(enumerate(shp), total=len(shp)):
            geom = shape(polygon['geometry'])
            idx.insert(id=i, coordinates=geom.bounds, obj=(polygon["properties"]["BRO_Geomor"], polygon['geometry']))
    print(idx.bounds)
    idx.close()


def test_idx():
    idx = index.Index("geomorph", interleaved=True)
    print(idx.bounds)
    ints = list(idx.intersection((70890.9455999993, 400263.8125, 70947.2773999982, 400366.219000001), objects="raw"))
    print(ints)


if __name__ == "__main__":
    shp_fn = "geomorph.shp"
    geomorph_to_idx(shp_fn)
    test_idx()
