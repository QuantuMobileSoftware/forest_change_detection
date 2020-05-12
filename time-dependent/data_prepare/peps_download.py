#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Source code copied from this repository. https://github.com/olivierhagolle/peps_download

This is a simple piece of code to automatically download the products provided by
the French Sentinel collaborative ground segment named PEPS : https://peps.cnes.fr.
PEPS is mirroring all the Sentinel data provided by ESA, and is providing a simplified access.

This code was written thanks to the precious help of one my colleagues at
CNES Jérôme Gasperi who developped the "rocket" interface which is used by Peps.

This code relies on python 2.7 (however, I just created a Python3 branch, which seems to work but was not much tested),
and on the curl utility. I am not sure it can work on windows.

Only the recent PEPS products or the frequently accessed ones are stored on disks (2 PB),
while the rest is stored on tapes (up to 14 PB). Data stored on tapes have an access time increased by 2 to 6 mn.
From the 23rd of March, peps_download has been fully reshaped to first stage products on tapes for download,
then download products on disk, which gives some time to upload the tape products on disks.
This procedures considerably speeds the downloads up.
"""
import json
import optparse
import os
import os.path
import sys
import time
from configparser import ConfigParser
from datetime import date, datetime, timedelta

import requests
from requests.auth import HTTPBasicAuth
from sentinelhub import WebFeatureService, BBox, CRS
from shapely.geometry import shape

config = ConfigParser(allow_no_value=True)
config.read('peps_download_config.ini')

def get_area_id(bbox, crs=CRS.WGS84):
    today = datetime.today()
    monthago = today - timedelta(days=7)
    time_interval = (
        monthago.strftime("%Y-%m-%dT%H:%M:%S"),
        today.strftime("%Y-%m-%dT%H:%M:%S")
    )

    search_bbox = BBox(bbox=bbox, crs=crs)
    wfs_iterator = WebFeatureService(
        search_bbox, time_interval,
        instance_id=config.get('config', 'sentinel_id')
    )
    aoi = search_bbox.get_geometry()
    instances = list(wfs_iterator)

    tiles = {}
    for i, tile_info in enumerate(wfs_iterator.get_tiles()):
        poly = shape(instances[i]['geometry'])
        tiles[tile_info[0]] = round(poly.intersection(aoi).area / poly.union(aoi).area * 100, 3)

    return sorted(tiles.items(), key=lambda x: -x[1])


###########################################################################


class OptionParser(optparse.OptionParser):

    def check_required(self, opt):
        option = self.get_option(opt)

        # Assumes the option's 'default' is set to None!
        if getattr(self.values, option.dest) is None:
            self.error("%s option not supplied" % option)


###########################################################################
def check_rename(tmpfile, prodsize, options):
    #print(os.path.getsize(tmpfile), prodsize)
    if os.path.getsize(tmpfile) != prodsize:
        with open(tmpfile) as f_tmp:
            try:
                tmp_data = json.load(f_tmp)
                raise Exception(f"Result is a json file (might come from a wrong password file)\ntemp data:{tmp_data}")
            except ValueError:
                print("\ndownload was not complete, tmp file removed")
                os.remove(tmpfile)
                pass
    else:
        os.rename("%s" % tmpfile, "%s/%s.zip" % (options.write_dir, prod))
        print("product saved as : %s/%s.zip" % (options.write_dir, prod))


###########################################################################


def parse_catalog(search_json_file):
    # Filter catalog result
    with open(search_json_file) as data_file:
        data = json.load(data_file)

    if 'ErrorCode' in data:
        raise Exception(data['ErrorMessage'])

    # Sort data
    download_dict = {}
    storage_dict = {}
    size_dict = {}
    if len(data["features"]) > 0:
        for i in range(len(data["features"])):
            prod = data["features"][i]["properties"]["productIdentifier"]
            #print(prod, data["features"][i]["properties"]["storage"]["mode"])
            feature_id = data["features"][i]["id"]
            try:
                storage = data["features"][i]["properties"]["storage"]["mode"]
                platform = data["features"][i]["properties"]["platform"]
                resourceSize = int(data["features"][i]["properties"]["resourceSize"])
                #print('resourceSize:', resourceSize)
                if storage == "unknown":
                    print('found a product with "unknown" status : %s' % prod)
                    print("product %s cannot be downloaded" % prod)
                    print('please send and email with product name to peps admin team : exppeps@cnes.fr')
                else:
                    # recup du numero d'orbite
                    orbitN = data["features"][i]["properties"]["orbitNumber"]
                    if platform == 'S1A':
                        # calcul de l'orbite relative pour Sentinel 1A
                        relativeOrbit = ((orbitN - 73) % 175) + 1
                    elif platform == 'S1B':
                        # calcul de l'orbite relative pour Sentinel 1B
                        relativeOrbit = ((orbitN - 27) % 175) + 1

                    if options.orbit is not None:
                        if platform.startswith('S2'):
                            if prod.find("_R%03d" % options.orbit) > 0:
                                download_dict[prod] = feature_id
                                storage_dict[prod] = storage
                                size_dict[prod] = resourceSize

                        elif platform.startswith('S1'):
                            if relativeOrbit == options.orbit:
                                download_dict[prod] = feature_id
                                storage_dict[prod] = storage
                                size_dict[prod] = resourceSize
                    else:
                        download_dict[prod] = feature_id
                        storage_dict[prod] = storage
                        size_dict[prod] = resourceSize

            except:
                pass

        '''
        # cloud cover criterium:
        if options.collection[0:2] == 'S2':
            for i in range(len(data["features"])):
                prod = data["features"][i]["properties"]["productIdentifier"]
                if data["features"][i]["properties"]["cloudCover"] > options.clouds:
                    del download_dict[prod], storage_dict[prod], size_dict[prod]
        '''
        # selecion of specific satellite
        if options.sat != None:
            for i in range(len(data["features"])):
                prod = data["features"][i]["properties"]["productIdentifier"]
                if data["features"][i]["properties"]["platform"] != options.sat:
                    try:
                        del download_dict[prod], storage_dict[prod], size_dict[prod]
                    except KeyError:
                        pass

        for prod in download_dict.keys():
            print(prod, storage_dict[prod])
    else:
        raise Exception(">>> no product corresponds to selection criteria")
    #    print(download_dict.keys())

    return (prod, download_dict, storage_dict, size_dict)


# ===================== MAIN
# ==================
# parse command line
# ==================
if len(sys.argv) == -1:
    prog = os.path.basename(sys.argv[0])
    print('      ' + sys.argv[0] + ' [options]')
    print("     Aide : ", prog, " --help")
    print("        ou : ", prog, " -h")
    print("example 1 : python %s -l 'Toulouse' -a peps.txt -d 2016-12-06 -f 2017-02-01 -c S2ST" %
          sys.argv[0])
    print("example 2 : python %s --lon 1 --lat 44 -a peps.txt -d 2015-11-01 -f 2015-12-01 -c S2" %
          sys.argv[0])
    print(
        "example 3 : python %s --lonmin 1 --lonmax 2 --latmin 43 --latmax 44 -a peps.txt -d 2015-11-01 -f 2015-12-01 -c S2" %
        sys.argv[0])
    print("example 4 : python %s -l 'Toulouse' -a peps.txt -c SpotWorldHeritage -p SPOT4 -d 2005-11-01 -f 2006-12-01" %
          sys.argv[0])
    print("example 5 : python %s -c S1 -p GRD -l 'Toulouse' -a peps.txt -d 2015-11-01 -f 2015-12-01" %
          sys.argv[0])
    sys.exit(-1)
else:
    usage = "usage: %prog [options] "
    parser = OptionParser(usage=usage)

    parser.add_option("-l", "--location", dest="location", action="store", type="string",
                      help="town name (pick one which is not too frequent to avoid confusions)",
                      default=config.get('config', 'location'))
    parser.add_option("-a", "--auth", dest="auth", action="store", type="string",
                      help="Peps account",
                      default=config.get('config', 'auth'))
    parser.add_option("--password", dest="password", action="store", type="string",
                      help="Peps password",
                      default=config.get('config', 'password'))
    parser.add_option("-w", "--write_dir", dest="write_dir", action="store", type="string",
                      help="Path where the products should be downloaded",
                      default=config.get('config', 'write_dir'))
    parser.add_option("-c", "--collection", dest="collection", action="store", type="choice",
                      help="Collection within theia collections", choices=['S1', 'S2', 'S2ST', 'S3'],
                      default=config.get('config', 'collection'))
    parser.add_option("-p", "--product_type", dest="product_type", action="store", type="string",
                      help="GRD, SLC, OCN (for S1) | S2MSI1C S2MSI2A S2MSI2Ap (for S2)",
                      default=config.get('config', 'product_type'))
    parser.add_option("-m", "--sensor_mode", dest="sensor_mode", action="store", type="string",
                      help="EW, IW , SM, WV (for S1) | INS-NOBS, INS-RAW (for S3)",
                      default=config.get('config', 'sensor_mode'))
    parser.add_option("-n", "--no_download", dest="no_download", action="store_true",
                      help="Do not download products, just print curl command",
                      default=config.getboolean('config', 'no_download'))
    parser.add_option("-d", "--start_date", dest="start_date", action="store", type="string",
                      help="start date, fmt('2015-12-22')",
                      default=config.get('config', 'start_date'))
    parser.add_option("-t", "--tile", dest="tile", action="store", type="string",
                      help="Sentinel-2 tile number",
                      default=config.get('config', 'tile'))
    parser.add_option("--lat", dest="lat", action="store", type="float",
                      help="latitude in decimal degrees",
                      default=config.get('config', 'lat'))
    parser.add_option("--lon", dest="lon", action="store", type="float",
                      help="longitude in decimal degrees",
                      default=config.get('config', 'lon'))
    parser.add_option("--latmin", dest="latmin", action="store", type="float",
                      help="min latitude in decimal degrees",
                      default=config.get('config', 'latmin'))
    parser.add_option("--latmax", dest="latmax", action="store", type="float",
                      help="max latitude in decimal degrees",
                      default=config.get('config', 'latmax'))
    parser.add_option("--lonmin", dest="lonmin", action="store", type="float",
                      help="min longitude in decimal degrees",
                      default=config.get('config', 'lonmin'))
    parser.add_option("--lonmax", dest="lonmax", action="store", type="float",
                      help="max longitude in decimal degrees",
                      default=config.get('config', 'lonmax'))
    parser.add_option("-o", "--orbit", dest="orbit", action="store", type="int",
                      help="Orbit Path number",
                      default=config.get('config', 'orbit'))
    parser.add_option("-f", "--end_date", dest="end_date", action="store", type="string",
                      help="end date, fmt('2015-12-23')",
                      default=config.get('config', 'end_date'))
    parser.add_option("--json", dest="search_json_file", action="store", type="string",
                      help="Output search JSON filename",
                      default=config.get('config', 'search_json_file'))
    parser.add_option("--windows", dest="windows", action="store_true",
                      help="For windows usage",
                      default=config.getboolean('config', 'windows'))
    parser.add_option("--cc", "--clouds", dest="clouds", action="store", type="int",
                      help="Maximum cloud coverage",
                      default=config.get('config', 'clouds'))
    parser.add_option("--sat", "--satellite", dest="sat", action="store", type="string",
                      help="S1A,S1B,S2A,S2B,S3A,S3B",
                      default=config.get('config', 'satellite'))
    (options, args) = parser.parse_args()

if options.search_json_file is None or options.search_json_file == "":
    options.search_json_file = 'search.json'

if options.sat != None:
    print(options.sat, options.collection[0:2])
    if not options.sat.startswith(options.collection[0:2]):
        raise Exception("input parameters collection and satellite are incompatible")

if options.tile is None:
    if options.location is None:
        if options.lat is None or options.lon is None:
            if (options.latmin is None) or (options.lonmin is None) or (options.latmax is None) or (
                    options.lonmax is None):
                raise Exception("provide at least a point or rectangle or tile number")
            else:
                geom = 'rectangle'
        else:
            if (options.latmin is None) and (options.lonmin is None) and (options.latmax is None) and (
                    options.lonmax is None):
                geom = 'point'
            else:
                raise Exception("please choose between point and rectangle, but not both")
    else:
        if (options.latmin is None) and (options.lonmin is None) and (options.latmax is None) and (
                options.lonmax is None) and (options.lat is None) or (options.lon is None):
            geom = 'location'
        else:
            raise Exception("please choose location and coordinates, but not both")

# geometric parameters of catalog request

if options.tile is not None:
    if options.tile.startswith('T') and len(options.tile) == 6:
        tileid = options.tile[1:6]
    elif len(options.tile) == 5:
        tileid = options.tile[0:5]
    else:
        raise Exception("tile name is ill-formated : 31TCJ or T31TCJ are allowed")
    query_geom = "tileid=%s" % (tileid)
elif geom == 'point':
    query_geom = 'lat=%f\&lon=%f' % (options.lat, options.lon)
elif geom == 'rectangle':
    query_geom = 'box={lonmin},{latmin},{lonmax},{latmax}'.format(
        latmin=options.latmin, latmax=options.latmax, lonmin=options.lonmin, lonmax=options.lonmax)
elif geom == 'location':
    query_geom = "q=%s" % options.location

# date parameters of catalog request
if options.start_date is not None:
    start_date = options.start_date
    if options.end_date is not None:
        end_date = options.end_date
    else:
        end_date = date.today().isoformat()

# special case for Sentinel-2

if options.collection == 'S2':
    if options.start_date >= '2016-12-05':
        print("**** products after '2016-12-05' are stored in Tiled products collection")
        print("**** please use option -c S2ST")
    elif options.end_date >= '2016-12-05':
        print("**** products after '2016-12-05' are stored in Tiled products collection")
        print("**** please use option -c S2ST to get the products after that date")
        print("**** products before that date will be downloaded")

if options.collection == 'S2ST':
    if options.end_date < '2016-12-05':
        print("**** products before '2016-12-05' are stored in non-tiled products collection")
        print("**** please use option -c S2")
    elif options.start_date < '2016-12-05':
        print("**** products before '2016-12-05' are stored in non-tiled products collection")
        print("**** please use option -c S2 to get the products before that date")
        print("**** products after that date will be downloaded")

# ====================
# read authentification
# ====================

email = options.auth
passwd = options.password
print(email)
print(passwd)
if os.path.exists(options.search_json_file):
    os.remove(options.search_json_file)

# ====================
# search in catalog
# ====================
# TODO(flyingpi): remove duplicated code.
search_catalog = f'https://peps.cnes.fr/resto/api/collections/{options.collection}/search.json?{query_geom}&startDate={start_date}&completionDate={end_date}&maxRecords=500'
if options.product_type or options.sensor_mode:
    search_catalog += f'&productType={options.product_type}&sensorMode={options.sensor_mode}'

print(search_catalog)
if options.windows:
    search_catalog = search_catalog.replace('\&', '^&')

print(search_catalog)
search_json = requests.get(search_catalog, allow_redirects=True)

if search_json.status_code >= 300:
    raise Exception(f'Can not load search json\n response message: {getattr(search_json, "content", "")}')

with open(options.search_json_file, 'wb') as f:
    f.write(search_json.content)

prod, download_dict, storage_dict, size_dict = parse_catalog(options.search_json_file)

# ====================
# Download
# ====================


if len(download_dict) == 0:
    print("No product matches the criteria")
else:
    # first try for the products on tape
    if options.write_dir == None:
        options.write_dir = os.getcwd()

    for prod in list(download_dict.keys()):
        file_exists = os.path.exists(("%s/%s.SAFE") % (options.write_dir, prod)
                                     ) or os.path.exists(("%s/%s.zip") % (options.write_dir, prod))
        if not (options.no_download) and not (file_exists):
            if storage_dict[prod] == "tape":
                tmticks = time.time()
                tmpfile = ("%s/tmp_%s.tmp") % (options.write_dir, tmticks)

                print("\nStage tape product: %s" % prod)
                product = requests.get(
                    f'https://peps.cnes.fr/resto/collections/{options.collection}/{download_dict[prod]}/download/?issuerId=peps',
                    allow_redirects=True,
                    auth=HTTPBasicAuth(email, passwd)
                )
                if product.status_code >= 300:
                    raise Exception(
                        f'can not load product from catalogue\n response message: {getattr(product, "content", "")}'
                    )

    NbProdsToDownload = len(list(download_dict.keys()))
    print("##########################")
    print("%d  products to download" % NbProdsToDownload)
    print("##########################")
    while (NbProdsToDownload > 0):
        # redo catalog search to update disk/tape status
        search_catalog = f'https://peps.cnes.fr/resto/api/collections/{options.collection}/search.json?{query_geom}&startDate={start_date}&completionDate={end_date}&maxRecords=500'
        if options.product_type == "" or options.sensor_mode == "":
            search_catalog += f'&productType={options.product_type}&sensorMode={options.sensor_mode}'

        if options.windows:
            search_catalog = search_catalog.replace('\&', '^&')
        search_json = requests.get(search_catalog, allow_redirects=True)
        if search_json.status_code >= 300:
            raise Exception(f'Can not load search json\n response message: {getattr(search_json, "content", "")}')
        with open(options.search_json_file, 'wb') as f:
            f.write(search_json.content)

        prod, download_dict, storage_dict, size_dict = parse_catalog(options.search_json_file)

        NbProdsToDownload = 0
        # download all products on disk
        for prod in list(download_dict.keys()):
            file_exists = os.path.exists(("%s/%s.SAFE") % (options.write_dir, prod)
                                         ) or os.path.exists(("%s/%s.zip") % (options.write_dir, prod))
            if (not (options.no_download) and not (file_exists)):
                if storage_dict[prod] == "disk":
                    tmticks = time.time()
                    tmpfile = ("%s/tmp_%s.tmp") % (options.write_dir, tmticks)
                    print("\nDownload of product : %s" % prod)
                    '''
		    get_product_url = f'https://peps.cnes.fr/resto/collections/{options.collection}/{download_dict[prod]}/download/?issuerId=peps'
                    print(get_product_url)

                    product = requests.get(get_product_url, allow_redirects=True, auth=HTTPBasicAuth(email, passwd))
                    if product.status_code < 300:
                        with open(tmpfile, 'wb') as f:
                            f.write(search_json.content)
                        check_rename(tmpfile, size_dict[prod], options)
                    else:
                        NbProdsToDownload += 1
                    '''
                    get_product = 'curl -o %s -k -u "%s:%s" https://peps.cnes.fr/resto/collections/%s/%s/download/?issuerId=peps' % (tmpfile, email, passwd, options.collection, download_dict[prod])
                    #print(get_product)
                    os.system(get_product)
                    # check binary product, rename tmp file
                    if not os.path.exists(("%s/tmp_%s.tmp") % (options.write_dir, tmticks)):
                        NbProdsToDownload += 1
                    else:
                        check_rename(tmpfile, size_dict[prod], options)
            elif file_exists:
                print("%s already exists" % prod)

        # download all products on tape
        for prod in list(download_dict.keys()):
            file_exists = os.path.exists(("%s/%s.SAFE") % (options.write_dir, prod)
                                         ) or os.path.exists(("%s/%s.zip") % (options.write_dir, prod))
            if (not (options.no_download) and not (file_exists)):
                if storage_dict[prod] == "tape" or storage_dict[prod] == "staging":
                    NbProdsToDownload += 1

        if NbProdsToDownload > 0:
            print("##############################################################################")
            print("%d remaining products are on tape, lets's wait 1 minute before trying again" %
                  NbProdsToDownload)
            print("##############################################################################")
            time.sleep(60)

if None not in [options.latmin, options.latmax, options.lonmin, options.lonmax]:
    print(
        '\nPreferred tile_id for specified area:',
        *get_area_id([options.lonmin, options.latmax, options.lonmax, options.latmin]),
        sep='\n'
    )
