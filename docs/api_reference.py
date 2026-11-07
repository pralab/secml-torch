"""Sphinx extension that builds the API reference from the ``secmlt`` package tree.

Nothing is listed by hand. Two directives are provided:

``.. secmlt-api:: public``
    One section per package that defines ``__all__``. Only the names exported in
    ``__all__`` are documented, under the short import path (e.g.
    ``secmlt.models.BasePyTorchClassifier``), in alphabetical order.

``.. secmlt-api:: internal``
    One section per non-package module with everything it defines, under the full
    module path. These entries are not indexed, so they never compete with the
    public names for cross-references.
"""

import importlib
import pkgutil
import types

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

ROOT = "secmlt"
EXCLUDED = {"tests"}


def _discover():
    """Return the public ``{package: exported names}`` and the internal module names.

    An object exported by several packages is listed only under the first one
    (in alphabetical order), so it is documented once.
    """
    root = importlib.import_module(ROOT)
    found, modules = {}, []
    for info in pkgutil.walk_packages(root.__path__, f"{ROOT}."):
        parts = info.name.split(".")[1:]
        if any(p.startswith("_") or p in EXCLUDED for p in parts):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception as exc:  # optional dependencies may be missing
            logger.info("[api] skipping %s (%s: %s)", info.name, type(exc).__name__, exc)
            continue
        if info.ispkg:
            if getattr(module, "__all__", None):
                found[info.name] = module
        else:
            modules.append(info.name)

    packages, seen = {}, set()
    for name in sorted(found):
        exported = []
        for attr in sorted(found[name].__all__):
            obj = getattr(found[name], attr, None)
            if isinstance(obj, (type, types.FunctionType)):
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
            exported.append((attr, obj))
        if exported:
            packages[name] = exported
    return packages, sorted(modules)


def _title(text):
    return f"{text}\n{'-' * len(text)}\n"


def _public_block(package, exported):
    """Document each exported name explicitly.

    ``automodule :members:`` only sees names in ``dir(module)``, so it would miss
    the names a package exports lazily through ``__getattr__``.
    """
    lines = [_title(package), f".. automodule:: {package}\n   :no-members:\n"]
    for attr, obj in exported:
        path = f"{package}.{attr}"
        if isinstance(obj, type):
            lines.append(f".. autoclass:: {path}\n   :members:\n   :show-inheritance:\n")
        elif isinstance(obj, types.FunctionType):
            lines.append(f".. autofunction:: {path}\n")
        else:
            # plain data directive: autodata would render the docstring of dict/int/str
            lines.append(f".. py:data:: {path}\n")
    return "\n".join(lines)


class SecmltApi(SphinxDirective):
    required_arguments = 1

    def run(self):
        kind = self.arguments[0]
        packages, modules = _discover()
        if kind == "public":
            blocks = [
                _public_block(name, exported) for name, exported in packages.items()
            ]
        elif kind == "internal":
            blocks = [
                f"{_title(name)}\n.. automodule:: {name}\n   :members:\n"
                "   :undoc-members:\n   :show-inheritance:\n   :no-index:\n"
                for name in modules
            ]
        else:
            raise self.error(f"secmlt-api: unknown kind {kind!r}")
        container = nodes.section()
        content = StringList("\n".join(blocks).splitlines(), source="secmlt-api")
        nested_parse_with_titles(self.state, content, container)
        return container.children


def setup(app):
    app.add_directive("secmlt-api", SecmltApi)
    return {"version": "0.1", "parallel_read_safe": False}
