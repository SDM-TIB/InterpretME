# InterpretME Example French Royalty Queries

The queries in this directory are instantiations of the template queries for _Louis XIV_ in the _French Royalty example_.
The queries return the following:

1. Retrieves everything that is known about the entity `http://interpretme.org/entity/Louis_XIV` in the InterpretME KG that was interpreted by LIME.
2. Shows the impact of the feature _childs_ for the classification of `http://interpretme.org/entity/Louis_XIV`.
3. Shows the impact of all features (that actually contribute, i.e., weight > 0.0) for the classification of `http://interpretme.org/entity/Louis_XIV`
4. Returns the results of the integrity constraint validation of `http://interpretme.org/entity/Louis_XIV`.
5. Retrieves the main characteristics of `http://interpretme.org/entity/Louis_XIV` from the input KG. The InterpretME KG contributes to this federated query by providing the URL of _Louix XIV_ in the input KG.
