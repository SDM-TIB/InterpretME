from rdfizer import semantify
import validating_models.stats as stats
from pkg_resources import resource_filename

semantification = stats.get_decorator('PIPE_InterpretMEKG_SEMANTIFICATION')


@semantification
def rdf_semantification():
    print("#####################################################################")
    print("******* Semantifying traced metadata from predictive pipeline *******")
    print("#####################################################################")
    main_path = resource_filename('InterpretME', '')
    config = '[default]\n' \
             'main_directory: ' + main_path + '\n\n' \
                                              '[datasets]\n' \
                                              'number_of_datasets: 11\n' \
                                              'output_folder: ./rdf-dump\n' \
                                              'all_in_one_file: yes\n' \
                                              'remove_duplicate: yes\n' \
                                              'name: interpretme\n' \
                                              'enrichment: yes\n' \
                                              'dbtype: mysql\n' \
                                              'ordered: yes\n' \
                                              'large_file: false\n\n' \
                                              '[dataset1]\n' \
                                              'name: classes\n' \
                                              'mapping: ${default:main_directory}/mappings/classes.ttl\n\n' \
                                              '[dataset2]\n' \
                                              'name: cross_validation\n' \
                                              'mapping: ${default:main_directory}/mappings/cross_validation.ttl\n\n' \
                                              '[dataset3]\n' \
                                              'name: endpoint\n' \
                                              'mapping: ${default:main_directory}/mappings/endpoint.ttl\n\n' \
                                              '[dataset4]\n' \
                                              'name: feature_definition\n' \
                                              'mapping: ${default:main_directory}/mappings/definition_feature.ttl\n\n' \
                                              '[dataset5]\n' \
                                              'name: imp_features\n' \
                                              'mapping: ${default:main_directory}/mappings/imp_features.ttl\n\n' \
                                              '[dataset6]\n' \
                                              'name: model_details\n' \
                                              'mapping: ${default:main_directory}/mappings/model_details.ttl\n\n' \
                                              '[dataset7]\n' \
                                              'name: precision_recall\n' \
                                              'mapping: ${default:main_directory}/mappings/precision_recall.ttl\n\n' \
                                              '[dataset8]\n' \
                                              'name: sampling_strategy\n' \
                                              'mapping: ${default:main_directory}/mappings/sampling_strategy.ttl\n\n' \
                                              '[dataset9]\n' \
                                              'name: shacl_validation_results\n' \
                                              'mapping: ${default:main_directory}/mappings/shacl_validation_results.ttl\n\n' \
                                              '[dataset10]\n' \
                                              'name: lime_features\n' \
                                              'mapping: ${default:main_directory}/mappings/lime_features.ttl\n\n' \
                                              '[dataset11]\n' \
                                              'name: prediction_probabilities\n' \
                                              'mapping: ${default:main_directory}/mappings/prediction_probabilities.ttl\n\n'

    with open('config_rdfizer.ini', 'w', encoding='utf8') as config_file:
        config_file.write(config)

    read_mappings = semantify("config_rdfizer.ini")
    print(read_mappings)
