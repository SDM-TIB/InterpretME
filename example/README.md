# InterpretME Example

You can run the pipeline with extended data from the French Royalty dataset by executing the following commands:

1. start the containers: `docker-compose up -d`
1. run the pipeline: `docker exec -it interpretme python Predictive_pipeline/extract_data.py --path example/example_french_royalty.json`
1. semantify the results: `docker exec -it sdmrdfizer /data/scripts/transform_load_archive.sh /data/config/rdfizer.ini`
