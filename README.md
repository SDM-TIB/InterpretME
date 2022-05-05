# InterpretME

This repository uses submodules, run `git clone --recurse-submodules git@github.com:SDM-TIB/InterpretME.git` to clone the repository including the submodules.

## The InterpretME Ontology
The ontology used to describe the metadata traced by InterpretMe can be explored in an instance of (WebVOWL)[http://ontology.tib.eu/InterpretME/visualization].

## Building the Docker image
If you want to build the Docker image yourself instead of pulling it from DockerHub, simply run: `docker build . -t sdmtib/interpretme:latest`


## Table specifying Classes and its Mapping rules
| Class | MappingRules | 
| ------ | ------ | 
| http://interpretme.org/vocab/SHACLValidation | 8 | 
| http://interpretme.org/vocab/PrecisionRecall | 7 | 
| http://interpretme.org/vocab/PredictionInterpretability | 6 | 
| http://interpretme.org/vocab/FeaturesWeights | 5 | 
| http://interpretme.org/vocab/TargetEntity | 5 | 
| http://interpretme.org/vocab/PredictionFeatures | 5 | 
| http://interpretme.org/vocab/FeatureDefinition | 4 | 
| http://www.w3.org/ns/mls#Implementation | 4 | 
| http://www.w3.org/ns/mls#Run | 3 | 
| http://www.w3.org/ns/mls#ModelEvaluation | 3 | 
| http://www.w3.org/ns/mls#HyperParameterSetting | 3 | 
| http://interpretme.org/vocab/TestedTargetEntity | 3 | 
| http://interpretme.org/vocab/PredictionClasses | 2 | 
| http://interpretme.org/vocab/SamplingStrategy | 2 |  
| http://interpretme.org/vocab/CrossValidation | 2 |  
| http://interpretme.org/vocab/Endpoint | 2 |  
| http://interpretme.org/vocab/ImportantFeature | 2 |  