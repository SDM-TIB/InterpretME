from InterpretME import pipeline, plots

results = pipeline(
    path_config='./example_french_royalty.json',
    lime_results='./interpretme/files/LIME',
    server_url='http://interpretmekg:8890/',
    username='dba',
    password='dba'
)

plots.sampling(results=results, path='./interpretme/output')
plots.feature_importance(results=results, path='./interpretme/output')
plots.decision_trees(results=results, path='./interpretme/output')
