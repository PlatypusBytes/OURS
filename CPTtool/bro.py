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
from os import stat, name
from zipfile import ZipFile
from tqdm import tqdm

# patch module-level attribute to enable pickle to work
# kdtree.node = kdtree.KDTree.node
# kdtree.leafnode = kdtree.KDTree.leafnode
# kdtree.innernode = kdtree.KDTree.innernode

fn = "/Volumes/wdmpu/bro/brocpt.xml"
searchstring = b"<gml:featureMember>"
columns = ["penetrationLength", "depth", "elapsedTime", "coneResistance", "correctedConeResistance", "netConeResistance", "magneticFieldStrengthX", "magneticFieldStrengthY", "magneticFieldStrengthZ", "magneticFieldStrengthTotal", "electricalConductivity",
           "inclinationEW", "inclinationNS", "inclinationX", "inclinationY", "inclinationResultant", "magneticInclination", "magneticDeclination", "localFriction", "poreRatio", "temperature", "porePressureU1", "porePressureU2", "porePressureU3", "frictionRatio"]
req_columns = ["penetrationLength", "depth", "coneResistance", "localFriction", "frictionRatio"]

ns = "{http://www.broservices.nl/xsd/cptcommon/1.1}"
ns2 = "{http://www.broservices.nl/xsd/dscpt/1.1}"
ns3 = "{http://www.opengis.net/gml/3.2}"
ns4 = "{http://www.broservices.nl/xsd/brocommon/3.0}"
ns5 = "{http://www.opengis.net/om/2.0}"
footer = b"</gml:FeatureCollection>"
nodata = -999999


def writexml(data):
    with open("test.xml", "wb") as f:
        f.write(data)


def parse_bro_xml(xml):
    root = etree.fromstring(xml)

    data = {"id": None, "location_x": None, "location_y": None,
            "offset_z": None, "predrilled_z": None}

    # Location
    for loc in root.iter(ns2 + "deliveredLocation"):
        for pos in loc.iter(ns3 + "pos"):
            x, y = pos.text.split(" ")
            data["location_x"] = float(x)
            data["location_y"] = float(y)

    # BRO Id
    for loc in root.iter(ns4 + "broId"):
        data["id"] = loc.text

    # NAP offset
    for loc in root.iter(ns + "offset"):
        z = loc.text
        data["offset_z"] = float(z)

    # Pre drilled depth
    for loc in root.iter(ns + "predrilledDepth"):
        z = loc.text
        data["predrilled_z"] = float(z)

    # Find which columns are not empty
    avail_columns = []
    for parameters in root.iter(ns + "parameters"):
        for parameter in parameters:
            if parameter.text == "ja":
                avail_columns.append(parameter.tag[len(ns):])

    # Determine if all data is available
    meta_usable = all([x is not None for x in data.values()])
    data_usable = all([col in avail_columns for col in req_columns])
    if not (meta_usable and data_usable):
        logging.warning("CPT misses required data.")
        return None

    # Parse data array, replace nodata, filter and sort
    for element in root.iter(ns + "values"):
        sar = StringIO(element.text.replace(";", "\n"))
        ar = np.loadtxt(sar, delimiter=",")
        ar[ar == nodata] = np.nan
        df = pd.DataFrame(ar, columns=columns)
        df = df[avail_columns]
        df.sort_values(by=['depth'], inplace=True)

    data["dataframe"] = df

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

    ext = splitext(fn)[1]

    # Setup progress
    pbar = tqdm(total=datasize//1000000, unit='Mbytes')

    # Memory map OS specifc options
    if name == 'nt':
        mm_options = {}


    if ext == ".xml":
        with open(fn, "r") as f:
            len_ss = len(searchstring)
            # memory-map the file, size 0 means whole file
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
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
                            pbar.update((i-previ)//1000000)
                            (x, y) = parse_xml_location(tdata)
                            locations.append((x, y, previ, i))
                    cpt_count += 1

    # This won't work yet!
    elif ext == ".zip":
        with ZipFile(fn) as zf:
            with zf.open("brocpt.xml") as f:
                with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
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
    else:
        raise Exception("Wrong database format.")

    pbar.close()

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
    tree = KDTree(npindex[:, 0:2])

    # Query point and return slices
    points = tree.query_ball_point((x, y), radius)

    # Return slices
    return npindex[points, 2:4].astype(np.int64)


def read_bro_xml(fn, indices):
    cpts = []
    with open(fn, "r") as f:
        # memory-map the file, size 0 means whole file
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
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
    indices = query_index(index, x, y, radius=parameters["Radius"])
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
    input = {"BRO_data": "./bro/brocpt.xml", "Source_x": 104882, "Source_y": 478455, "Radius": 1000}

    cpts = read_bro(input)
    print(cpts)
