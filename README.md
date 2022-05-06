[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# InterpretME

![InterpretME Architecture](/images/architecture.png "InterpretME Architecture")

InterpretME integrates knowledge graphs with machine learning methods to generate interesting meaningful insights. 
It helps to generate human- and machine-readable decisions to provide assistance to users and enhance efficiency.
InterpretME is a tool for fine-grained representations, in a knowledge graph, of the main characteristics of trained machine learning models. 

## The InterpretME Ontology
The ontology used to describe the metadata traced by InterpretMe can be explored in an instance of [WebVOWL](http://ontology.tib.eu/InterpretME/visualization).
The table below describes the number of mapping rules per class.

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
