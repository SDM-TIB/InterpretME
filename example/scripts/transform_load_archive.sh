#!/bin/bash

python3 /data/scripts/transform_and_load.py -c $1
/data/scripts/archive.sh


