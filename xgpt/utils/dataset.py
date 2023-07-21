import importlib


def local_dataset_module(dm_module, dataset_name, dataset_type="datasets"):
    parent_modules = dm_module.split(".")[:-1]
    module_path = ".".join(parent_modules + [dataset_type, dataset_name])
    module = importlib.import_module(module_path)

    return module
