#!/usr/bin/env python3

import os  

try:
	virtuosoIP = os.environ["SPARQL_ENDPOINT_IP"]
	virtuosoUser = os.environ["SPARQL_ENDPOINT_USER"]
	virtuosoPass = os.environ["SPARQL_ENDPOINT_PASSWD"]
	virtuosoPort = os.environ["SPARQL_ENDPOINT_PORT"]
	virtuosoGraph = os.environ["SPARQL_ENDPOINT_GRAPH"]
	outputfolder = os.environ["RDF_DUMP_FOLDER_PATH"]
except Exception as e:
	print("Exception while loading RDF data to viruoso  endpoint: ", e)
	exit(-1)

os.system("/data/scripts/virtuoso-script.sh " + virtuosoIP + " " + virtuosoUser + " " + virtuosoPass + " " + virtuosoPort + " " + virtuosoGraph + " " + outputfolder)