from InterpretME import pipeline

pipeline(
    path_config='./example_french_royalty.json',
    lime_results='./interpretme/files/LIME',
    server_url='http://interpretmekg:8890/',
    username='dba',
    password='dba'
)
