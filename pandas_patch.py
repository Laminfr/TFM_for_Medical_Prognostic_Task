"""
Pandas Compatibility Patch (Why we need it)

The package `auton_survival` depends on older versions of `pandas` and `scikit-survival`, and it fails to import correctly on modern Python environments (Python ≥3.10 and Pandas ≥2.0).
-> We cannot install 'auton_survival' via pip because:

* It tries to compile old versions of `scikit-survival` that are incompatible with modern Python versions.
* It enforces strict version pins (`numpy <2.0`, `pandas <2.0`, etc.)
* Build fails with Cython and wheel errors during `scikit-survival` compilation.

Instead of using pip, clone the repository locally:

pip install git+https://github.com/IBM/AutoNSurvival.git
cd auton-survival

Then open 'pyproject.toml' file and update the dependencies to versions that work with our modern environment like:

python = "^3.8"
torch = "^2.1"
numpy = "^1.24"
pandas = "^2.0"
tqdm = "^4.66"
scikit-learn = "^1.6"
torchvision = "^0.16"
scikit-survival = "^0.24"
lifelines = "^0.26"

Several functions inside `lifelines` still rely on deprecated pandas methods such as:

- Series.iteritems()
- DataFrame .describe(datetime_is_numeric=...)

These methods were removed or changed in recent pandas releases.
To avoid breaking code in the entire project, we include a small compatibility patch, which:

1. Restores `Series.iteritems` by redirecting it to `.items`
2. Wraps DataFrame `.describe()` so unsupported arguments are dropped
3. Silences deprecated warnings that make the console unreadable

This patch is implemented in this file and should be imported instead of pandas:

from pandas_patch import pd

This way, all files that use pandas (pd) automatically benefit from the fix.
"""
import sys
import pandas as _pd
if not hasattr(_pd.Series, "iteritems"):
    _pd.Series.iteritems = _pd.Series.items
_old_describe = _pd.DataFrame.describe
def _describe_compat(self, *args, **kwargs):
    kwargs.pop("datetime_is_numeric", None)
    return _old_describe(self, *args, **kwargs)
_pd.DataFrame.describe = _describe_compat
pd = _pd
sys.modules['pandas'] = pd