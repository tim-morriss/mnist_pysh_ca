"""pysh_ca."""
import collections
import collections.abc

# ponytail: pyshgp==0.1.8 (latest release) does `from collections import Sequence`,
# removed in Python 3.10. Restore the alias here so pyshgp keeps working without
# forking/patching the dependency. Drop this if pyshgp ever fixes it upstream.
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

__version__ = "0.0.1"
