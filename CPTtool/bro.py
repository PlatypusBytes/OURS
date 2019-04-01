#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BRO XML CPT database reader, indexer and parser.

Enables searching of very large XML CPT database dumps.
In order to speed up these operations, an index will
be created by searching for `featureMember`s (CPTs) in the xml
if it does not yet exist next to the XML file.

The index is stored next to the file and stores the xml
filesize to validate the xml database is the same. If not,
we assumme the database is new and a new index will be created
as well. The index itself is an array with columns that store
the x y location of the CPT and the start/end bytes in the XML file.

As of January 2019, almost a 100.000 CPTs are stored in the XML
and creating the index will take 5-10min depending on disk performance.

"""

import sys
import mmap
import logging
from io import StringIO
import pickle
from os.path import exists, splitext
from os import stat, name
from zipfile import ZipFile

# External modules
from lxml import etree
import numpy as np
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
import pandas as pd
import pyproj

# Constants for XML parsing
searchstring = b"<gml:featureMember>"
footer = b"</gml:FeatureCollection>"

columns = ["penetrationLength", "depth", "elapsedTime", "coneResistance", "correctedConeResistance", "netConeResistance", "magneticFieldStrengthX", "magneticFieldStrengthY", "magneticFieldStrengthZ", "magneticFieldStrengthTotal", "electricalConductivity",
           "inclinationEW", "inclinationNS", "inclinationX", "inclinationY", "inclinationResultant", "magneticInclination", "magneticDeclination", "localFriction", "poreRatio", "temperature", "porePressureU1", "porePressureU2", "porePressureU3", "frictionRatio"]
req_columns = ["penetrationLength", "depth", "coneResistance", "localFriction", "frictionRatio"]

ns = "{http://www.broservices.nl/xsd/cptcommon/1.1}"
ns2 = "{http://www.broservices.nl/xsd/dscpt/1.1}"
ns3 = "{http://www.opengis.net/gml/3.2}"
ns4 = "{http://www.broservices.nl/xsd/brocommon/3.0}"
ns5 = "{http://www.opengis.net/om/2.0}"

nodata = -999999
to_epsg = "28992"
to_srs = pyproj.Proj(init='epsg:{}'.format(to_epsg))


def writexml(data, id="test"):
    """Quick function to write xml in memory to disk.
    :param data: XML bytes.
    :param id: Filename to use."""
    with open("{}.xml".format(id), "wb") as f:
        f.write(data)


def parse_bro_xml(xml):
    """Parse bro CPT xml.
    Searches for the cpt data, but also
    - location
    - offset z
    - id
    - predrilled_z
    TODO Replace iter by single search
    as iter can give multiple results

    :param xml: XML bytes
    :returns: dict -- parsed CPT data + metadata
    """
    root = etree.fromstring(xml)

    # Initialize data dictionary
    data = {"id": None, "location_x": None, "location_y": None,
            "offset_z": None, "predrilled_z": None}

    # Location
    x, y = parse_xml_location(xml)
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
        data["predrilled_z"] = z

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
    for cpt in root.iter(ns + "conePenetrationTest"):
        for element in cpt.iter(ns + "values"):
            # Load string data and parse as 2d array
            sar = StringIO(element.text.replace(";", "\n"))
            ar = np.loadtxt(sar, delimiter=",")

            # Check shape of array
            found_columns = ar.shape[1]
            if found_columns != len(columns):
                writexml(xml, id=data["id"])
                logging.warning("Data has the wrong size! {} columns instead of {}".format(found_columns, len(columns)))
                return None

            # Replace nodata constant with nan
            # Create a DataFrame from array
            # and sort by depth
            ar[ar == nodata] = np.nan
            df = pd.DataFrame(ar, columns=columns)
            df = df[avail_columns]
            df.sort_values(by=['depth'], inplace=True)

        data["dataframe"] = df

    return data


def parse_xml_location(tdata):
    """Return x y of location.
    TODO Don't user iter
    :param tdata: XML bytes
    :returns: list -- of x y string coordinates

    Will transform coordinates not in EPSG:28992
    """
    root = etree.fromstring(tdata)
    crs = None

    for loc in root.iter(ns2 + "deliveredLocation"):
        for point in loc.iter(ns3 + "Point"):
            srs = point.get("srsName")
            if srs is not None and "EPSG" in srs:
                crs = srs.split("::")[-1]
            break
        for pos in loc.iter(ns3 + "pos"):
            x, y = map(float, pos.text.split(" "))
            break

    if crs is not None and crs != to_epsg:
        logging.warning("Reprojecting from epsg::{}".format(crs))
        source_srs = pyproj.Proj('+init=epsg:{}'.format(crs))
        x, y = pyproj.transform(source_srs, to_srs, y, x)
        return x, y
    else:
        return x, y


def create_index(fn, ifn, datasize):
    """Create an index into the large BRO xml database.

    :param fn: Filename for bro xml file.
    :param ifn: Filename for index of fn.
    :param datasize: int -- Size of bro xml file.
    :returns: list -- of locations and indices into file.
    """

    logging.warning("Creating index, this may take a while...")

    # Iterate over file to search for Features
    locations = []
    cpt_count = 0

    ext = splitext(fn)[1]

    # Setup progress
    pbar = tqdm(total=datasize, unit_scale=1)

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
                    i = mm.find(searchstring, i + 1)
                    data = mm[previ:i]

                    if cpt_count == 0:
                        header = data
                        footer = b"</gml:FeatureCollection>"
                    else:
                        tdata = header + data + footer
                        if i != -1:
                            pbar.update((i - previ))

                            (x, y) = parse_xml_location(tdata)
                            locations.append((x, y, previ, i))
                    cpt_count += 1

    # This won't work yet!
    # TODO Implement zip streaming
    elif ext == ".zip":
        with ZipFile(fn) as zf:
            with zf.open("brocpt.xml") as f:
                with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
                    i = 0
                    while i != -1:
                        previ = i
                        i = mm.find(searchstring, i + 1)
                        data = mm[previ:i]

                        if cpt_count == 0:
                            header = data
                            footer = b"</gml:FeatureCollection>"
                        else:
                            tdata = header + data + footer
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


def query_index(index, x, y, radius=1000.):
    """Query database for CPTs
    within radius of x, y.

    :param index: Index is a array with columns: x y begin end
    :type index: np.array
    :param x: X coordinate
    :type x: float
    :param y: Y coordinate
    :type y: float
    :param radius: Radius (m) to use for searching. Defaults to 1000.
    :type radius: float
    :return: 2d array of start/end (columns) for each location (rows).
    :rtype: np.array
    """

    # Setup KDTree based on points
    npindex = np.array(index)
    tree = KDTree(npindex[:, 0:2])

    # Query point and return slices
    points = tree.query_ball_point((x, y), radius)

    # Return slices
    return npindex[points, 2:4].astype(np.int64)


def read_bro_xml(fn, indices):
    """Read XML file at specific indices and parse these.

    :param fn: Bro XML filename.
    :type fn: str
    :param indices: List of tuples containing start/end bytes.
    :type indices: list
    :return: List of parsed CPTs as dicts
    :rtype: list

    """
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
    """Main function to read the BRO database.

    :param parameters: Dict of input `parameters` containing filename, locationd and radius.
    :type parameters: dict
    :return: List of parsed CPTs as dicts
    :rtype: list

    """
    fn = parameters["BRO_data"]
    ifn = splitext(fn)[0] + ".idx"  # index
    x, y = parameters["Source_x"], parameters["Source_y"]

    if not exists(fn):
        print("Cannot open provided BRO data file: {}".format(fn))
        sys.exit(2)

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
    input = {"BRO_data": "./bro/brocpt.xml", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
    cpts = read_bro(input)
    print(cpts)
