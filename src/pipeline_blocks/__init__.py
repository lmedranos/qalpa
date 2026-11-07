"""Pipeline processing blocks with lazy imports for optional dependencies."""

from importlib import import_module

__all__ = [
    "relax_molecule",
    "MolecularFilter",
    "sascore",
    "calc_properties",
    "preprocess_fhi",
    "prop_postprocess",
    "run_to_hdf5",
    "create_EquiDTB_json",
    "create_FHI_json",
]

_EXPORTS = {
    "relax_molecule": (".optimize_geometry", "relax_molecule"),
    "MolecularFilter": (".filter_molecules", "MolecularFilter"),
    "sascore": (".calculate_sascore", "sascore"),
    "calc_properties": (".prop_calc", "calc_properties"),
    "preprocess_fhi": (".propcalc_preprocess", "preprocess_fhi"),
    "prop_postprocess": (".propcalc_postprocess", "prop_postprocess"),
    "run_to_hdf5": (".to_hdf5", "run_to_hdf5"),
    "create_EquiDTB_json": (".parserEquiDTBtoJSON", "get_data"),
    "create_FHI_json": (".parserFHItoJSON", "get_data"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
