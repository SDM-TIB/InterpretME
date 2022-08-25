# InterpretME: A Tool for Interpretations of Machine Learning Models Over Knowledge Graphs
InterpretME is a publicly available library.
It includes a pipeline for enhancing the interpretability of machine learning models over knowledge graphs,
an ontology to describe the main characteristics of trained machine learning models, and the InterpretME knowledge graph.
The InterpretME KG assists the users in clarification and ease the interpretation of the model's predictions of a particular entity aligned with SHACL validation results.
InterpretME uses state-of-the-art machine learning models and interpretable tools.
InterpretME evaluates the SHACL constraints over the nodes of the input KGs and generates a validation report per constraint and target entity.
It helps the user to understand the decisions of the predictive models and also provides a platform for interpretability.

## Installation

InterpretME is OS independent, i.e., you can run it on Linux, Mac, and Windows.
However, the current version only supports Python 3.8 and 3.9.
You can install InterpretME from PyPI via pip:
```sh
pip install InterpretME
```

## Running the InterpretME Pipeline
```python
from InterpretME import pipeline
pipeline(path_config, sampling, cv, imp_features, test_split, model, lime_results)
```

`pipeline()` executes the whole pipeline; including extracting data and metadata from the input KGs, validating SHACL constraints, preprocessing the data and running predictive models.
InterpretME aims at collecting metadata at each step of pipeline.
The current version of InterpretME resorts to interpretable surrogate tools like LIME [1].
The user can provide a path to store the LIME results.
Even model performance metrics like accuracy, precision etc. are recorded as metadata.
The RDF mapping language (RML) is used to define mappings for the metadata collected from the predictive pipeline in order to integrate them into the **InterpretME KG**.
The RML mappings are used by the SDM-RDFizer [2], an efficient RML engine for creating knowledge graphs, to semantify the metadata.
The function `pipeline()` returns results from the pipeline which are used later in traceability of a target entity.

**Parameters:**
- `path_config` - Path to the input configuration file (JSON) for Input KG
- `sampling` - Sampling strategy to use (undersampling or oversampling)
- `cv` - Number of cross-validation folds required while performing stratified shuffle split
- `imp_features` - Number of important features
- `test_split` - Splitting of training and testing dataset
- `model` - Model used to perform stratified shuffle split (Random forest, Adaboost classifier, Gradient boosting classifier)
- `lime_results` - Path to store LIME results in HTML format

**Returns:**
A dictionary that captures all the results of the trained predictive model stored as objects which can be used for further analysis for e.g., `plots.sampling()`.

## Plots
InterpretME offers plots to understand and visualize the characteristics of the trained predictive model.
The following plot functions are defined in InterpretME.

### Sampling
```python
from InterpretME import plots
plots.sampling(results, path)
```
`plots.sampling()` saves the target class distribution plot after applying the sampling strategy.

**Parameters:**
- `results` - Results dictionary obtained from `pipeline()`
- `path` - Path where to store the output plot

### Feature Importance
```python
from InterpretME import plots
plots.feature_importance(results,path)
```
`plots.feature_importance()` Creates a bar plot of important features with feature weights. 

**Parameters:**
- `results` - Results dictionary obtained from `pipeline()`
- `path` - Path where to store the output plot

### Decision Trees
```python
from InterpretME import plots
plots.decision_trees(results,path)
```
`plots.decision_trees()` plots the decision trees generated from the predictive model.

**Parameters:**
- `results` - Results dictionary obtained from `pipeline()`
- `path` - Path where to store the output plot

### Decision Trees with Constraint Validation
```python
from InterpretME import plots
plots.constraints_decision_trees(results,path)
```
`plot.constraints_decision_trees()` plots decision trees including SHACL constraint validation results.

**Parameters:**
- `results` - Results dictionary obtained from `pipeline()`
- `path` - Path where to store the output plot

## Federated Querying
InterpretME assists the user in interpreting the predictive model via its ability to trace all characteristics of a target entity from the input KG and InterpretME KG.
This is achieved by using the federated query engine *Detrusty* [3]. Using this module, the user's questions can be answered via SPARQL queries.

### Configuration
```python
from InterpretME.federated_query_engine import configuration
configuration(interpretme_endpoint, input_endpoint)
```
DeTrusty relies on collected metadata about the KGs.
`configuration()` collects the required metadata and stores it in a file as well as returning it. 

**Parameters:**
- `interpretme_endpoint` - URL of the InterpretME KG
- `input_endpoint` - URL of the input KG

**Returns:**
An instance of `DeTrusty.Molecule.MTManager.Config` that holds the metadata collected from the input KG and the InterpretME KG.
This object is to be used for the parameter `config` of the method `federated()`.

### Querying
```python
from InterpretME.federated_query_engine import federated
federated(input_query, config)
```

**Parameters:**
- `input_query` - SPARQL query to answer the user's question
- `config` - The configuration object holding the metadata about the KGs to query (generated by `configuration()`)

**Returns:**
A Python dictionary following the SPARQL protocol with the query result.

---
***References***

[1] Marco Ribeiro, Sameer Singh, and Carlos Guestrin. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). ACM. 2016. DOI: [10.1145/2939672.2939778](https://dl.acm.org/doi/10.1145/2939672.2939778).

[2] E. Iglesias, S. Jozashoori, D. Chaves-Fraga, D. Collarana and M.-E. Vidal. SDM-RDFizer: An RML Interpreter for the Efficient Creation of RDF Knowledge Graphs. In: CIKM ’20:Proceedings of the 29th ACM International Conference on Information & Knowledge Management, ACM, New York, NY,USA, 2020. DOI: [10.1145/3340531.3412881](https://dl.acm.org/doi/pdf/10.1145/3340531.3412881).

[3] P.D. Rohde. DeTrusty v0.6.1, August 2022. DOI: [10.5281/zenodo.6998001](https://doi.org/10.5281/zenodo.6998001).
