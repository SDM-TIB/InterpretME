import os
import sys

from InterpretME import pipeline, plots

path_config = str(sys.argv[1])
if not os.path.isfile(path_config):
    print('No valid configuration path given! Falling back to default: ./example_kg_french_royalty.json')
    path_config = './example_kg_french_royalty.json'

results = pipeline(
    path_config=path_config,
    lime_results='./interpretme/files/LIME',
    shap_results='./interpretme/files/SHAP',
    server_url='http://interpretmekg:8890/',
    username='dba',
    password='dba'
)

plots.sampling(results=results, path='./interpretme/output')
plots.feature_importance(results=results, path='./interpretme/output')
plots.decision_trees(results=results, path='./interpretme/output')
plots.constraints_decision_trees(results=results, path='./interpretme/output', constraint_num=[1, 2, 3])
