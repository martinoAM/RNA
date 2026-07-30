import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type
_original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _patched_get_type(schema):
    """Version tolérante de get_type qui gère les schémas booléens."""
    if isinstance(schema, bool):
        return "Any"
    return _original_get_type(schema)


def _patched_json_schema_to_python_type(schema, defs=None):
    """Version tolérante de _json_schema_to_python_type (gère les
    schémas booléens rencontrés dans les clés comme 'additionalProperties')."""
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)


_gc_utils.get_type = _patched_get_type
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
