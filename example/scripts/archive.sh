#!/bin/bash

cd /data/interpretme
echo "Archiving run..."
mkdir -p archive  # no error if archive already exists
run_id=$( head -2 files/classes.csv  | tail -1 | cut -d ',' -f1 )
GZIP=-9 tar cvzf ./archive/run_${run_id}.tar.gz ./files/ ./dataset/ ./output/
rm -rf ./output ./files ./dataset
echo "Done archiving!"

