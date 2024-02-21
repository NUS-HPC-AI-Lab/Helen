import glob
import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

py_list = glob.glob(os.path.join(BASE_DIR, "*.py"))
__all__ = []
for f in py_list:
    f_name = os.path.basename(f)
    model_name = f_name.split(".")[0]
    if not model_name.startswith("_"):
        try:
            import_command = f"from .{model_name} import {model_name}"
            exec(import_command)
            __all__.append(model_name)
        except Exception as e:
            pass
