# BRO Datadump reader and parser
import mmap
import logging
from lxml import etree
import numpy as np
from io import StringIO
# from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree as KDTree
import pickle
import pandas as pd
from os.path import exists, splitext
from os import stat

# patch module-level attribute to enable pickle to work
# kdtree.node = kdtree.KDTree.node
# kdtree.leafnode = kdtree.KDTree.leafnode
# kdtree.innernode = kdtree.KDTree.innernode

fn = "/Volumes/wdmpu/bro/brocpt.xml"
searchstring = b"<gml:featureMember>"
columns = ["penetrationLength", "depth", "elapsedTime", "coneResistance", "correctedConeResistance", "netConeResistance", "magneticFieldStrengthX", "magneticFieldStrengthY", "magneticFieldStrengthZ", "magneticFieldStrengthTotal", "electricalConductivity",
           "inclinationEW", "inclinationNS", "inclinationX", "inclinationY", "inclinationResultant", "magneticInclination", "magneticDeclination", "localFriction", "poreRatio", "temperature", "porePressureU1", "porePressureU2", "porePressureU3", "frictionRatio"]

ns = "{http://www.broservices.nl/xsd/cptcommon/1.1}"
ns2 = "{http://www.broservices.nl/xsd/dscpt/1.1}"
ns3 = "{http://www.opengis.net/gml/3.2}"
footer = b"</gml:FeatureCollection>"

def writexml(data, i, header, footer):
    with open("test_{}.xml".format(cpt_count + 1), "wb") as f:
        f.write(header)
        f.write(data)
        f.write(footer)


def parse_bro_xml(xml):
    root = etree.fromstring(xml)

    for loc in root.iter(ns2 + "deliveredLocation"):
        for pos in loc.iter(ns3 + "pos"):
            return pos.text.split(" ")

    return
    allowed_params = []
    for parameters in root.iter(ns + "parameters"):
        for parameter in parameters:
            if parameter.text == "ja":
                allowed_params.append(parameter.tag[len(ns):])
    for element in root.iter(ns + "values"):
        sar = StringIO(element.text.replace(";", "\n"))
        ar = np.loadtxt(sar, delimiter=",")
        ar[ar == -999999] = np.nan
        # print(ar.shape, ar)
        df = pd.DataFrame(ar, columns=columns)
        df = df[allowed_params]
        df.sort_values(by=['depth'], inplace=True)
        # print(df)
        df.to_csv("test_{}.csv".format(cpt_count))
    writexml(data, i, header, footer)
    return data


def parse_xml_location(tdata):
    """Return x y of location."""
    root = etree.fromstring(tdata)

    for loc in root.iter(ns2 + "deliveredLocation"):
        for pos in loc.iter(ns3 + "pos"):
            return pos.text.split(" ")


def create_index(fn, ifn, datasize):
    logging.warning("Creating index, this may take a while...")

    # Iterate over file to search for Features
    locations = []
    cpt_count = 0

    with open(fn, "r") as f:
        len_ss = len(searchstring)
        # memory-map the file, size 0 means whole file
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        i = 0
        while i != -1:
            previ = i
            i = mm.find(searchstring, i+1)
            data = mm[previ:i]

            if cpt_count == 0:
                header = data
                footer = b"</gml:FeatureCollection>"
            else:
                tdata = header+data+footer
                if i != -1:
                    (x, y) = parse_xml_location(tdata)
                    locations.append((x, y, previ, i))
            cpt_count += 1
        mm.close()

    # Write size and locations to file
    with open(ifn, "wb") as f:
        pickle.dump((datasize, locations), f)

    # But return index for immediate use
    return locations


def query_index(index, x, y, radius=1000):
    """Query database for CPTs
    within radius of x, y.

    Index is a array with columns: x y begin end"""

    # Setup KDTree based on points
    npindex = np.array(index)
    print(npindex)
    tree = KDTree(npindex[:, 0:2])

    # Query point and return slices
    points = tree.query_ball_point((x, y), radius)

    # Return slices
    return npindex[points, 2:4].astype(int)


def read_bro_xml(fn, indices):
    cpts = []
    with open(fn, "r") as f:
        # memory-map the file, size 0 means whole file
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        i = mm.find(searchstring, 0)
        header = mm[0:i]
        for (start, end) in indices:
            data = mm[start:end]
            tdata = header + data + footer
            cpt = parse_bro_xml(tdata)
            cpts.append(cpt)
        mm.close()
    return cpts


def read_bro(parameters):
    fn = parameters["BRO_data"]
    ifn = splitext(fn)[0] + ".idx"  # index
    x, y = parameters["Source_x"], parameters["Source_y"]

    if not exists(fn):
        raise Exception("Cannot open provided BRO data file: {}".format(fn))

    # Check and use/create index
    datasize = stat(fn).st_size
    if exists(ifn):
        with open(ifn, "rb") as f:
            (size, index) = pickle.load(f)
        if size != datasize:
            logging.warning("BRO datafile differs from index, recreating index.")
            index = create_index(fn, ifn, datasize)
    else:
        index = create_index(fn, ifn, datasize)

    # Find CPT indexes
    indices = query_index(index, x, y)
    n_cpts = len(indices)
    if n_cpts == 0:
        logging.warning("Found no CPTs, try another location or increase the radius.")
        return []
    else:
        logging.info("Found {} CPTs".format(len(indices)))

    # Open database and retrieve CPTs
    # TODO Open zipfile instead of large xml
    cpts = read_bro_xml(fn, indices)

    return cpts


if __name__ == "__main__":
    input = {"BRO_data": "/Volumes/wdmpu/bro/brocpt.xml", "Source_x": 104882, "Source_y": 478455}
    cpts = read_bro(input)
    print(cpts)
