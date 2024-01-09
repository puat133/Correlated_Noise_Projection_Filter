import h5py
from typing import Dict, List, Any
import jax.numpy as jnp
import numpy as np
import pathlib


def save_to_hdf(file_name, variable_dict: Dict, classes: list = [], excluded_vars: List = None):
    """
    Save variable in a dictionary to a hdf file

    Parameters
    ----------
    file_name: string or pathlib.Path
        file name of the hdf file
    variable_dict: dict
        Dictionary to be saved
    classes: List
        List of classes
    excluded_vars: List
        Excluded variables

    Returns
    -------
    None
    """

    try:
        with h5py.File(file_name, 'w') as f:
            __save_object(f, variable_dict, classes, excluded_vars=excluded_vars)
    except AttributeError as e:
        print("Something wrong here ", e.__class__, " occured.")
        print("Full detail ", e.__cause__)


def __save_object(f, obj, classes: list = [], end_here=False, excluded_vars=None):
    if excluded_vars is None:
        excluded_vars = []
    types = [int, float, str, bool]
    if isinstance(obj, dict):
        dict_ = obj
    elif isinstance(obj, list):
        name_ = [str(i_) for i_ in range(len(obj))]
        dict_ = dict(zip(name_,
                         obj))
    else:
        dict_ = obj.__dict__

    for key, value in dict_.items():
        if key in excluded_vars:
            continue
        if __is_instance(value, types):
            f.create_dataset(key, data=value)
            continue
        elif __is_instance(value, [jnp.ndarray, np.ndarray]):
            if value.dtype in [jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64, jnp.complex128,
                               np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]:
                if value.ndim > 0:
                    f.create_dataset(key, data=value, compression='gzip')
                else:
                    f.create_dataset(key, data=value)
                continue
        else:
            if not end_here:
                if __is_instance(value, classes):
                    grp = f.create_group(key)
                    __save_object(grp, value, excluded_vars=excluded_vars)


def __is_instance(value: Any, classes: List):
    it_is = False
    for a_type in classes:
        it_is = isinstance(value, a_type)
        if it_is:
            break
    return it_is


def __is_list_of_strings_contains_a_string(a_string: str, list_of_strings: List):
    it_is = False
    for another_string in list_of_strings:
        it_is = (another_string in a_string)
        if it_is:
            break
    return it_is


def get_variables(file: h5py.File, data_set_names: List, array_names: List) -> Dict:
    variables = {}
    for i in range(len(data_set_names)):
        try:
            variables[array_names[i]] = file[data_set_names[i]][()]
        except KeyError:
            print("Warning, dataset with name = {} does not exists in this file!".format(data_set_names[i]))
            variables[array_names[i]] = None
    return variables


def load_hdf_file(file_name: str,
                  relative_path: str,
                  data_set_names: List,
                  array_names: List) -> Dict:
    simulation_results_path = pathlib.Path(relative_path)
    if not simulation_results_path.exists():
        raise FileNotFoundError

    file = simulation_results_path / file_name
    with h5py.File(file, mode='r') as f:
        array_dict = get_variables(f, data_set_names, array_names)
    return array_dict
