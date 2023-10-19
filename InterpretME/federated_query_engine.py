import re

import pandas as pd
from DeTrusty import run_query
from DeTrusty.Molecule.MTCreation import create_rdfmts
from DeTrusty.Molecule.MTManager import Config
from pkg_resources import resource_filename

re_service = re.compile(r".*[^:][Ss][Ee][Rr][Vv][Ii][Cc][Ee]\s*<.+>\s*{.*", flags=re.DOTALL)


def configuration(interpretme_endpoint, input_endpoint):
    """
    Creates the source description needed for DeTrusty to federate queries. The source description
    is created for the federation containing both the InterpretME KG and the input KG.

    Parameters
    ----------
    interpretme_endpoint : str
        URL of the InterpretME knowledge graph's SPARQL endpoint.
    input_endpoint : str
        URL of the input knowledge graph's SPARQL endpoint.

    Returns
    -------
    Config
        An instance of ConfigFile holding the source descriptions of the federation
        containing the InterpretME KG and the input KG.

    """
    main_path = resource_filename('InterpretME', '')

    endpoints_dict = {
        interpretme_endpoint: {
            'mappings': [
                str(main_path) + '/mappings/classes.ttl',
                str(main_path) + '/mappings/endpoint.ttl',
                str(main_path) + '/mappings/cross_validation.ttl',
                str(main_path) + '/mappings/definition_feature.ttl',
                str(main_path) + '/mappings/imp_features.ttl',
                str(main_path) + '/mappings/lime_features.ttl',
                str(main_path) + '/mappings/model_details.ttl',
                str(main_path) + '/mappings/precision_recall.ttl',
                str(main_path) + '/mappings/prediction_probabilities.ttl',
                str(main_path) + '/mappings/sampling_strategy.ttl',
                str(main_path) + '/mappings/entity_alignment.ttl',
                str(main_path) + '/mappings/shacl_validation_results.ttl'
            ]
        },
        input_endpoint: {}
    }

    return create_rdfmts(endpoints_dict, None)


def federated(query, configuration=Config()):
    """
    Executes a SPARQL query using the federated query engine DeTrusty.

    Parameters
    ----------
    query : str
        Input query.
    configuration : ConfigFile, OPTIONAL
        The ConfigFile object holding the source descriptions for the federated query engine.
        If no source description is provided, the federated query engine will not be able
        to execute a query without SERVICE clause.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe which contains the query result.

    """
    service = True if re_service.match(query) else False
    query_result = run_query(query, sparql_one_dot_one=service, config=configuration, join_stars_locally=False)

    columns = query_result['head']['vars']
    df_result = pd.DataFrame(columns=columns)
    cardinality = 0
    for res in query_result['results']['bindings']:
        df_result.loc[cardinality] = [res[var]['value'] for var in columns]
        cardinality += 1

    return df_result


if __name__ == "__main__":
    input_query = """  """

    interpretme_endpoint = ''
    input_endpoint = ''

    config = configuration(interpretme_endpoint, input_endpoint)
    federated(input_query, config)
