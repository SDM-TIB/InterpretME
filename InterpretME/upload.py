import requests
from requests.auth import HTTPBasicAuth
import time
import validating_models.stats as stats

uploading = stats.get_decorator('PIPE_InterpretMEKG_UPLOAD_VIRTUOSO')

@uploading
def upload_to_virtuoso(run_id, rdf_file, server_url, username, password):
    """

    Parameters
    ----------
    run_id : int
        Unique identifier.
    rdf_file : str
        Path to the RDF file.
    server_url : str
        InterpretME knowledge graph endpoint.
    username : str
        Username of virtuoso.
    password : str
        Password of virtuoso.

    Returns
    -------

    """
    # server_url does not end with /sparql
    print("#####################################################################")
    print("******************* Uploading to Virtuoso **************************")
    print("#####################################################################")
    address = server_url + 'DAV/home/' + username + '/rdf_sink/' + str(run_id) + '.ttl'
    with open(rdf_file, 'r') as f:
        r = requests.put(
            address,
            auth=HTTPBasicAuth(username, password),
            data=f.read(),
            headers={'Content-Type': 'text/turtle'}
        )
        if r.status_code != 201:
            raise Exception('Something went wrong!' + str(r.status_code))


