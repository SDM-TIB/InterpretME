# InterpretME Example

You can run the pipeline with extended data from the French Royalty dataset by executing the following commands:

1. start the containers: `docker-compose up -d`
1. run the pipeline: `docker exec -it interpretme python Predictive_pipeline/pipeline.py --path example/example_french_royalty.json`
1. semantify the results: `docker exec -it sdmrdfizer /data/scripts/transform_load_archive.sh /data/config/rdfizer.ini`

After running the pipeline and uploading the data into the SPARQL endpoint, DeTrusty can be used via its API to answer queries.
In `queries/templates` you can find query templates (placeholders marked with `$$`) for answering the following questions:

1. Which is the target entity interpreted by LIME?
1. How does _feature_ contribute to the classification of this entity in class _class_
1. Which other features are relevant for this classification?
1. Does this target entity satisfy the domain integrity constraints?
1. What are the main characteristics of the target entity?

In the folder `queries/french_royalty` you can find the query templates populated for the French Royalty dataset.
Use the following command to execute the first query for the French Royalty dataset:
```bash
curl -X POST -d "query=$(cat queries/french_royalty/Q1.sparql)" -d "sparql1_1=True" localhost:5000/sparql
```

For more information about DeTrusty, check its (GitHub repository)[https://github.com/SDM-TIB/DeTrusty].

