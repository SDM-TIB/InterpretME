from DeTrusty.Molecule.MTCreation import create_rdfmts
from DeTrusty.Molecule.MTManager import ConfigFile
from DeTrusty import run_query
from pkg_resources import resource_filename
import re

re_service = re.compile(r".*[^:][Ss][Ee][Rr][Vv][Ii][Cc][Ee]\s*<.+>\s*{.*", flags=re.DOTALL)


def configuration(interpretme_endpoint, input_endpoint):
    """

    Parameters
    ----------
    interpretme_endpoint : str
        InterpretME knowledge graph endpoint.
    input_endpoint : str
        Input knowledge graph endpoint.

    Returns
    -------
    config

    """
    main_path = resource_filename('InterpretME', '')
    # print(main_path)
    mappings = [str(main_path) + '/mappings/classes.ttl',
                str(main_path) + '/mappings/endpoint.ttl',
                str(main_path) + '/mappings/cross_validation.ttl',
                str(main_path) + '/mappings/definition_feature.ttl',
                str(main_path) + '/mappings/imp_features.ttl',
                str(main_path) + '/mappings/lime_features.ttl',
                str(main_path) + '/mappings/model_details.ttl',
                str(main_path) + '/mappings/precision_recall.ttl',
                str(main_path) + '/mappings/prediction_probabilities.ttl',
                str(main_path) + '/mappings/sampling_strategy.ttl',
                str(main_path) + '/mappings/shacl_validation_results.ttl']

    endpoints_dict = {
        interpretme_endpoint: {
            'mappings': mappings
        },
        input_endpoint: {}
    }

    create_rdfmts(endpoints_dict, './rdfmts.json')
    return ConfigFile('./rdfmts.json')


def federated(query, configuration):
    """

    Parameters
    ----------
    query : str
        Input query.
    configuration : config

    Returns
    -------

    """
    print("#####################################################################")
    print("********************* Federated Query Engine ************************")
    print("#####################################################################")
    service = True if re_service.match(query) else False
    query_result = run_query(query, sparql_one_dot_one=service, config=configuration)
    print(query_result)
    return query_result

if __name__ == "__main__":
    input_query = """  """

    interpretme_endpoint = ''
    input_endpoint = ''

    config = configuration(interpretme_endpoint, input_endpoint)
    federated(input_query, config)