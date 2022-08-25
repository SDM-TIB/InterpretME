[![DOI](https://zenodo.org/badge/488505724.svg)](https://zenodo.org/badge/latestdoi/488505724)
[![Docker Image](https://img.shields.io/badge/Docker%20Image-sdmtib/interpretme-blue?logo=Docker)](https://hub.docker.com/r/sdmtib/interpretme)
[![Latest Release](http://img.shields.io/github/release/SDM-TIB/InterpretME.svg?logo=github)](https://github.com/SDM-TIB/InterpretME/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# InterpretME

![InterpretME Architecture](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/architecture.png "InterpretME Architecture")

InterpretME integrates knowledge graphs (KG) with machine learning methods to generate interesting meaningful insights. 
It helps to generate human- and machine-readable decisions to provide assistance to users and enhance efficiency.
InterpretME is a tool for fine-grained representations, in a KG, of the main characteristics of trained machine learning models. 
It receives as input the features' definition, classes and the SHACL constraints from multiple KGs.
InterpretME takes JSON input from the user as shown below. The features' definition are classified into independent and dependent variables later used in the predictive models.
The feature definition has the following format _"x": "?x a <http://dbpedia.org/ontology/Person>. \n ", "gender": "Optional { ?x <http://dbpedia.org/ontology/gender> ?gender.}_ where the first part states the attribute from the KG and the later part describes the definition of that attribute in the KG using SPARQL.
This definition of features allows InterpretME to trace back the origin of that feature in the KG.
Given the features' definitions and the target definition, a _SELECT_ SPARQL query is built to retrieve the application domain data. 
InterpretME also takes constraints as input from the user to check if the entity validates or invalidates the constraints.
InterpretME is divided into two main quadrants.
The first one is "Training interpretable predictive model" and the second is "Documenting interpretable predictive model".
In brief, the first quadrant is responsible to perform all the predictive model pipeline components which include data preparation, applying sampling strategy to the data, building the predictive model and lastly generating visualization of the predictive models encompassed with the SHACL constraints.
The second quadrant "Documenting of interpretable predictive model" provides assistance to the user by generating the InterpretME KG and executing federated query on top of the InterpretME KG and original KG.
This, in turn, helps user to perform data exploration and trace the entity predicted with all the relevant features in the original KG.
Additionally, different metrics like precision, recall and accuracy along with LIME interpretations are provided to the user.

```json
{
    "Endpoint": "http://frenchroyalty:8890/sparql",
    "Type": "Person",
    "Index_var": "x",
    "Independent_variable": {
      "x": "?x a <http://dbpedia.org/ontology/Person>. \n ",
      "gender": "Optional { ?x <http://dbpedia.org/ontology/gender> ?gender } .\n ",
      "childs": "?x <http://dbpedia.org/ontology/numChilds> ?childs . \n ",
      "predecessors": "?x <http://dbpedia.org/ontology/numPredecessors> ?predecessors . \n",
      "preds": "?x <http://dbpedia.org/ontology/numPreds> ?preds .\n",
      "objects": "?x <http://dbpedia.org/ontology/numObjects> ?objects . \n",
      "subjects": "?x <http://dbpedia.org/ontology/numSubjects> ?subjects . \n"
    },
    "Dependent_variable": {
        "HasSpouse": "{ SELECT ?x, ((?partners > 0) AS ?HasSpouse) WHERE { ?x <http://dbpedia.org/ontology/numSpouses> ?partners . }} \n"
    },
    "Constraints": [
      {
        "name": "C3",
        "inverted": false,
        "shape_schema_dir": "example/shapes/french_royalty/spouse/rule3",
        "target_shape": "Spouse"
      },
      {
        "name": "C2",
        "inverted": false,
        "shape_schema_dir": "example/shapes/french_royalty/spouse/rule2",
        "target_shape": "Spouse"
      },
      {
        "name": "C1",
        "inverted": false,
        "shape_schema_dir": "example/shapes/french_royalty/spouse/rule1",
        "target_shape": "Spouse"
      }

    ],
    "classes": {
      "NoSpouse": "0",
      "HasSpouse": "1"
    },
    "3_valued_logic": true,
    "sampling_strategy": "undersampling",
    "number_important_features": 5,
    "cross_validation_folds": 5
}
```

## The InterpretME Ontology
The ontology used to describe the metadata traced by InterpretME can be explored in [VoCoL](http://ontology.tib.eu/InterpretME) and [WebProtégé](https://webprotege.stanford.edu/#projects/4dfe5ddb-752e-4dc9-b360-943785f0b0af/edit/Classes) (WebProtégé account required).

![InterpretME Ontology Visualization](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/ontology_vis.png "InterpretME Ontology Visualization")

The table below describes the number of mapping rules per class. You can find the mappings in `InterpretME/mappings` or query them only in a [public SPARQL endpoint](https://labs.tib.eu/sdm/InterpretME-mappings/sparql).

| Class                                                   | MappingRules |
|---------------------------------------------------------|--------------|
| http://interpretme.org/vocab/SHACLValidation            | 8            |
| http://interpretme.org/vocab/PrecisionRecall            | 7            |
| http://interpretme.org/vocab/PredictionInterpretability | 6            |
| http://interpretme.org/vocab/FeaturesWeights            | 5            |
| http://interpretme.org/vocab/TargetEntity               | 5            |
| http://interpretme.org/vocab/PredictionFeatures         | 5            |
| http://interpretme.org/vocab/FeatureDefinition          | 4            |
| http://www.w3.org/ns/mls#Implementation                 | 4            |
| http://www.w3.org/ns/mls#Run                            | 3            |
| http://www.w3.org/ns/mls#ModelEvaluation                | 3            |
| http://www.w3.org/ns/mls#HyperParameterSetting          | 3            |
| http://interpretme.org/vocab/TestedTargetEntity         | 3            |
| http://interpretme.org/vocab/PredictionClasses          | 2            |
| http://interpretme.org/vocab/SamplingStrategy           | 2            |
| http://interpretme.org/vocab/CrossValidation            | 2            |
| http://interpretme.org/vocab/Endpoint                   | 2            |
| http://interpretme.org/vocab/ImportantFeature           | 2            |


## Experiment Results
We were running experiments with InterpretME over an extended version of the French Royalty KG [1] (see `example/data`).
The task was to predict whether a person in the dataset has a spouse.
We perform under-sampling for this experiment to balance the two classes.

![DT Result](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/DT_final_results.png "DT Result")

The above figure shows the decision tree for the predictive task over the data.

![DT with Constraint Validation](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/constraints_validation_dtree.png "DT with Constraint Validation")

Since InterpretME uses SHACL constraints to validate the model, we can also include the validation results in the visualization.
In this case, the target entities fulfilled all the constraints or the constraints did not apply for the classification.

![Random Forest Feature Importance](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/Random_Forest_Feature_Importance.png "Random Forest Feature Importance")

The above figure shows the list of relevant features in random forest; most important on top, following features with decreasing importance.

![Target Entity Degree Distribution](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/DegreeDistribution.png "Target Entity Degree Distribution")

The average number of neighbours in the original KG was 11.39 (std 5.06).
With the metadata traced by InterpretME, the number increased to 26.99 (std 6.94).
The increase in the average number of neighbours shows that InterpretME enhances the interpretability of the target entities.
The original KG is available as a [public SPARQL endpoint](https://labs.tib.eu/sdm/InterpretME-og/sparql).
The original data enhanced with the metadata traced by InterpretME is also publicly available as a [SPARQL endpoint](https://labs.tib.eu/sdm/InterpretME-wog/sparql).

**References**

[1] Marco Ribeiro, Sameer Singh, and Carlos Guestrin. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*. ACM. 2016. DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)

## Running InterpretME
### Building InterpretME from Source
This repository uses submodules, please execute the following command to ensure all source files are cloned:
```bash
git clone --recurse-submodules git@github.com:SDM-TIB/InterpretME.git
```

After cloning the repository and changing into the repository directory, you can build the Docker image
```bash
docker build . -t sdmtib/interpretme:latest
```

Follow the instructions in the `example` directory for further information on how to proceed.

### Using existing Resources
If you are not interested in building InterpretME from source, you can simply follow the instruction in the `example` directory.
All steps necessary to run the pipeline, upload the data to a SPARQL endpoint, and query the InterpretME KG are described there.

## License
This work is licensed under the MIT license.

## Authors
InterpretME has been developed by members of the Scientific Data Management Group at TIB, as an ongoing research effort.
The development is co-ordinated and supervised by Maria-Esther Vidal.
We strongly encourage you to report any issues you have with InterpretME.
Please, use the GitHub issue tracker to do so.
InterpretME has been implemented in joint work by Yashrajsinh Chudasama, Disha Purohit, Julian Gercke, and Philipp D. Rohde.
