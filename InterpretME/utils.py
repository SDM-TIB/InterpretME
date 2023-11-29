import os
import sys

from rdflib import Graph
from validating_models.constraint import ShaclSchemaConstraint

PREFIX_InterpretME = 'http://interpretme.org/entity/'

QUERY_SHAPES = """
SELECT ?s WHERE {
  {
    ?s a <http://www.w3.org/ns/shacl#NodeShape>
    FILTER(isIRI(?s))
  }
  UNION
  {
    ?s a <http://www.w3.org/ns/shacl#PropertyShape>
    FILTER(isIRI(?s))
  }
}
"""

QUERY_UPDATE_SUBJECT = """
DELETE {{
  <{old}> ?p ?o .
}}
INSERT {{
  <{new}> ?p ?o .
}}
WHERE {{
  <{old}> ?p ?o .
}}
"""

QUERY_UPDATE_OBJECT = """
DELETE {{
  ?s ?p <{old}> .
}}
INSERT {{
  ?s ?p <{new}> .
}}
WHERE {{
  ?s ?p <{old}> .
}}
"""

QUERY_UPDATE_SHAPE_SCHEMA = """
DELETE {{}}
INSERT {{ <{schema}> <http://interpretme.org/vocab/hasShape> <{shape}> . }}
WHERE {{}}
"""


def _rename_shape(shape_uri_old, shapes_graph, schema_name, run_id):
    shape_name = shapes_graph.qname(shape_uri_old)
    if ':' in shape_name:
        shape_name = shape_name.split(':')[1]
    return PREFIX_InterpretME + str(run_id) + '_' + schema_name + '_' + shape_name


def load_and_rename_shapes(schema_dir, target_shape, schema_name, run_id):
    shapes_graph = Graph()
    for shape_file in os.listdir(schema_dir):
        shapes_graph.parse(os.path.join(schema_dir, shape_file))

    for shape in shapes_graph.query(QUERY_SHAPES):
        shape_uri_old = str(shape[0])
        shape_uri_new = _rename_shape(shape_uri_old, shapes_graph, schema_name, run_id)

        shapes_graph.update(QUERY_UPDATE_SUBJECT.format(old=shape_uri_old, new=shape_uri_new))
        shapes_graph.update(QUERY_UPDATE_OBJECT.format(old=shape_uri_old, new=shape_uri_new))

    return shapes_graph, '<' + _rename_shape(target_shape[1:-1], shapes_graph, schema_name, run_id) + '>'


def get_constraints(input_data_constraints, run_id):
    constraints = []
    for constraint in input_data_constraints:
        constraint['shape_schema_dir'], constraint['target_shape'] = load_and_rename_shapes(constraint['shape_schema_dir'], constraint['target_shape'], constraint['name'], run_id)
        constraints.append(ShaclSchemaConstraint.from_dict(constraint))
    return constraints


def update_schema(shapes_graph, constraint_id, run_id):
    query = 'SELECT ?s WHERE { ?s a <http://www.w3.org/ns/shacl#NodeShape> . FILTER(isIRI(?s)) }'
    schema = 'http://interpretme.org/entity/{schema}_{run_id}'.format(schema=constraint_id, run_id=run_id)
    for shape in shapes_graph.query(query):
        shapes_graph.update(QUERY_UPDATE_SHAPE_SCHEMA.format(schema=schema, shape=str(shape[0])))
    return shapes_graph


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

global pbar
