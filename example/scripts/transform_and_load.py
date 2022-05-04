#!/usr/bin/env python3

import os  
import sys
import logging
import traceback
import getopt
from configparser import ConfigParser, ExtendedInterpolation

from rdfizer.semantify import semantify

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()

def transform(configfile, script="/data/scripts/virtuoso-script.sh"):
	config = ConfigParser(interpolation=ExtendedInterpolation())
	config.read(configfile)
	try:
		logger.info("Transforming data using " + str(configfile) + " configuration...")
		outputfolder = config["datasets"]["output_folder"]
		semantify(configfile)
		status = os.path.exists(outputfolder)
		if status:
			virtuosoIP = os.environ["SPARQL_ENDPOINT_IP"]
			virtuosoUser = os.environ["SPARQL_ENDPOINT_USER"]
			virtuosoPass = os.environ["SPARQL_ENDPOINT_PASSWD"]
			virtuosoPort = os.environ["SPARQL_ENDPOINT_PORT"]
			virtuosoGraph = os.environ["SPARQL_ENDPOINT_GRAPH"]
			outputfolder = os.environ["RDF_DUMP_FOLDER_PATH"]
			try:
				os.system( str(script) + " " + virtuosoIP + " " + virtuosoUser + " " + virtuosoPass + " " + virtuosoPort + " " + virtuosoGraph + " " + outputfolder)
				logger.info("Semantification sucessful!")
			except Exception as ex:
				logger.error("ERROR while loading data to viruoso! " + str(ex))
				exc_type, exc_value, exc_traceback = sys.exc_info()
				emsg = repr(traceback.format_exception(exc_type, exc_value, exc_traceback))
				logger.error("Exception while semantifying ... " + str(emsg))

		else:
			logger.error("Error during semantification of data. Please check your configureation file.")
	    
	except Exception as e:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		emsg = repr(traceback.format_exception(exc_type, exc_value,
	                                           exc_traceback))
		logger.error("Exception while semantifying ... " + str(emsg))
	  

def get_options(argv):
	try:
		opts, args = getopt.getopt(argv, "h:c:s:")
	except getopt.GetoptError:
		usage()
		sys.exit(1)

	configfile = None
	script = "/data/scripts/virtuoso-script.sh"
    
	for opt, arg in opts:
		if opt == "-h":
			usage()
			sys.exit()
		elif opt == "-c":
			configfile = arg       
		elif opt == "-s":
			script = arg

	if not configfile:
		usage()
		sys.exit(1)

	return (configfile, script)

def usage():
	usage_str = ("Usage: {program} -c <pathto config.ini file>  -s <path to virtuoso script file> \n")
	print (usage_str.format(program=sys.argv[0]),)


def main(argv):
    
	configfile, script = get_options(argv[1:])
	try:
 		transform(configfile, script)
	except Exception as ex:
		print(ex)


if __name__ == '__main__':
	main(sys.argv)