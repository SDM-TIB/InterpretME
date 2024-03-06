import validating_models.stats as stats
from pkg_resources import resource_filename
from rdfizer import semantify

from InterpretME.utils import HiddenPrints

semantification = stats.get_decorator('PIPE_InterpretMEKG_SEMANTIFICATION')


@semantification
def rdf_semantification(input_is_kg: bool):
    config = {
        'default': {
            'main_directory': resource_filename('InterpretME', '')
        },
        'datasets': {
            'number_of_datasets': '13' if input_is_kg else '9',
            'output_folder': './rdf-dump',
            'all_in_one_file': 'yes',
            'remove_duplicate': 'yes',
            'name': 'interpretme',
            'enrichment': 'yes',
            'ordered': 'yes',
            'large_file': 'false'
        },
        'dataset1': {
            'name': 'classes',
            'mapping': '${default:main_directory}/mappings/classes.ttl'
        },
        'dataset2': {
            'name': 'cross_validation',
            'mapping': '${default:main_directory}/mappings/cross_validation.ttl'
        },
        'dataset3': {
            'name': 'imp_features',
            'mapping': '${default:main_directory}/mappings/imp_features.ttl'
        },
        'dataset4': {
            'name': 'model_details',
            'mapping': '${default:main_directory}/mappings/model_details.ttl'
        },
        'dataset5': {
            'name': 'precision_recall',
            'mapping': '${default:main_directory}/mappings/precision_recall.ttl'
        },
        'dataset6': {
            'name': 'sampling_strategy',
            'mapping': '${default:main_directory}/mappings/sampling_strategy.ttl'
        },
        'dataset7': {
            'name': 'lime_features',
            'mapping': '${default:main_directory}/mappings/lime_features.ttl'
        },
        'dataset8': {
            'name': 'prediction_probabilities',
            'mapping': '${default:main_directory}/mappings/prediction_probabilities.ttl'
        },
        'dataset9': {
            'name': 'shap_importance',
            'mapping': '${default:main_directory}/mappings/shap_importance.ttl'
        }
    }

    if input_is_kg:
        config['dataset10'] = {
            'name': 'endpoint',
            'mapping': '${default:main_directory}/mappings/endpoint.ttl'
        }
        config['dataset11'] = {
            'name': 'feature_definition',
            'mapping': '${default:main_directory}/mappings/definition_feature.ttl'
        }
        config['dataset12'] = {
            'name': 'shacl_validation_results',
            'mapping': '${default:main_directory}/mappings/shacl_validation_results.ttl'
        }
        config['dataset13'] = {
            'name': 'entity_alignment',
            'mapping': '${default:main_directory}/mappings/entity_alignment.ttl'
        }

    with HiddenPrints():
        semantify(config)
