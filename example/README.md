# InterpretME Example

You can run the pipeline with extended data from the French Royalty dataset by executing the following commands:

1. start the containers: `docker-compose up -d`
2. run the pipeline: `docker exec -it interpretme bash -c "cd example; python example_pipeline.py <config_json>"` where`<config_json>` is one of `example_kg_french_royalty.json` and `example_csv_french_royalty.json`.

After running the pipeline, the produced plots can be found in `./interpretme/output/`.
Once the data is uploaded into the SPARQL endpoint, DeTrusty can be used to answer queries.
In `queries/templates` you can find query templates (placeholders marked with `$$`) for answering the following questions:

1. Which is the target entity interpreted by LIME?
2. How does _feature_ contribute to the classification of this entity in class _class_
3. Which other features are relevant for this classification?
4. Does this target entity satisfy the domain integrity constraints?
5. What are the main characteristics of the target entity?

In the folder `queries/french_royalty` you can find the query templates populated for the French Royalty dataset.
Since InterpretME integrates DeTrusty, you can use the following command to execute the first query for the French Royalty dataset:
```bash
docker exec -it interpretme bash -c "cd example; python example_query.py queries/french_royalty/Q1.sparql"
```

For more information about DeTrusty, check its [GitHub repository](https://github.com/SDM-TIB/DeTrusty).

## IPython Notebook
There is also an IPython Notebook in this folder called `InterpretME_french_royalty.ipynb` that demonstrates the use of InterpretME.
