# InterpretME

This repository uses submodules, run `git clone --recurse-submodules git@github.com:SDM-TIB/InterpretME.git` to clone the repository including the submodules.

![InterpretME architecture](https://raw.githubusercontent.com/SDM-TIB/InterpretME/main/images/architecture.png "InterpretME architecture")

InterpretME integrates knowledge graphs with machine learning methods to generate interesting meaningful insights. 
It helps to generate human and machine readable decisions to provide assistance to users and enhance efficiency.
InterpretME a tool for fine-grained representations, in a knowledge graph, of the main characteristics of trained machine learning models. 

## Building the Docker image
If you want to build the Docker image yourself instead of pulling it from DockerHub, simply run: `docker build . -t sdmtib/interpretme:latest`

## Example
For an example on how to use InterpretME and benefit from the traced metadata, check the `example` folder.

## The InterpretME Ontology
The ontology used to describe the metadata traced by InterpretMe can be explored in an instance of [WebVOWL](http://ontology.tib.eu/InterpretME/visualization).
The table below describes the number of mapping rules per class.
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
------------------------------------------------------ 